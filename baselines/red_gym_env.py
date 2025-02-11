import uuid
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
from pyboy.logger import log_level
import mediapy as media
from einops import repeat
from skimage.transform import resize
import hnswlib

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from rich import print
import os

event_flags_start = 0xD747
event_flags_end = 0xD7F6 # 0xD761 # 0xD886 temporarily lower event flag range for obs input
museum_ticket = (0xD754, 0)

class RedGymEnv(Env):
    def __init__(self, config=None):
        
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = config["frame_stacks"]
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.restricted_start_menu = False if 'restricted_start_menu' not in config else config['restricted_start_menu']
        self.explore_weight = 1 if "explore_weight" not in config else config["explore_weight"]
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.randomize_first_ep_split_cnt = 0 if 'randomize_first_ep_split_cnt' not in config else config['randomize_first_ep_split_cnt']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        self.n_pokemon_features = 23
        self.vec_dim = 4096
        self.num_elements = 20000 # max
        self.output_shape = (36, 40)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.frame_stacks,
            self.output_shape[0],
            self.output_shape[1]
        )
        self.output_vector_shape = (54, )

        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A]
        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]
        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                # WindowEvent.PASS
            ])

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]
        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        print(os.getcwd())
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Discrete( n = 600, start = 0),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks),
                "seen_pokemon": spaces.MultiBinary(152),
                "caught_pokemon": spaces.MultiBinary(152),
                "moves_obtained": spaces.MultiBinary(0xA5),
            }
        )

        head = "headless" if config["headless"] else "SDL2"

        log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            debugging=False,
            disable_input=False,
            window_type=head,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)
    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)
        
    def reset(self, seed=None):
        self.seed = seed
        
        self.init_map_mem()
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
            self.output_shape[1]),
            dtype=np.uint8)


        self.agent_stats = []

        self.explore_map_dim = 384
        self.explore_map = np.zeros((self.explore_map_dim,self.explore_map_dim), dtype=np.uint8)

        self.recent_screens = np.zeros( self.output_shape, dtype=np.uint8)
        
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        self.visited_pokecenter_list = []
        self.visited_pokecenter = 0
        self.init_caches() 

        self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}
    def update_frame_knn_index(self, frame_vec):
        
        # if self.get_levels_sum() >= 22 and not self.levels_satisfied:
        #     self.levels_satisfied = True
        #     self.base_explore = self.knn_index.get_current_count()
        #     self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )
    def get_all_event_ids_obs(self):
        # max 249
        # padding_idx = 0
        # change dtype to uint8 to save space
        return np.array(self.last_10_event_ids[:, 0] + 1, dtype=np.uint8)
    
    def get_all_event_step_since_obs(self):
        step_gotten = self.last_10_event_ids[:, 1]  # shape (10,)
        step_since = self.step_count - step_gotten
        # step_count - step_since and scaled_encoding
        return self.scaled_encoding(step_since, 1000).reshape(-1, 1)  # shape (10,)

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray()[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    def boey_render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = game_pixels_render[:, :, 0]  # should be 3x speed up for rendering
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                reduced_frame = game_pixels_render
                self.recent_frames[0] = reduced_frame
            if add_memory:
                # pad = np.zeros(
                #     shape=(self.mem_padding, self.output_shape[1], 3), 
                #     dtype=np.uint8)
                # game_pixels_render = np.concatenate(
                #     (
                #         self.create_exploration_memory(), 
                #         pad,
                #         self.create_recent_memory(),
                #         pad,
                #         rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                #     ),
                #     axis=0)
                game_pixels_render = {
                    'image': self.recent_frames,
                    'vector': self.get_all_raw_obs(),
                    'map_ids': self.get_last_map_id_obs(),
                    'item_ids': self.get_all_item_ids_obs(),
                    'item_quantity': self.get_items_quantity_obs(),
                    'poke_ids': self.get_all_pokemon_ids_obs(),
                    'poke_type_ids': self.get_all_pokemon_types_obs(),
                    'poke_move_ids': self.get_all_move_ids_obs(),
                    'poke_move_pps': self.get_all_move_pps_obs(),
                    'poke_all': self.get_all_pokemon_obs(),
                    'event_ids': self.get_all_event_ids_obs(),
                    'event_step_since': self.get_all_event_step_since_obs(),
                    # 'in_battle_mask': self.get_in_battle_mask_obs(),
                }

        return game_pixels_render
    def get_knn_reward(self, last_event_rew):
        if last_event_rew != self.max_event_rew:
            # event reward increased, reset exploration
            if self.use_screen_explore:
                self.prev_knn_rew += self.knn_index.get_current_count()
                self.knn_index.clear_index()
            else:
                self.prev_knn_rew += len(self.seen_coords)
                self.seen_coords = {}
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        return (self.prev_knn_rew + cur_size) * self.explore_weight * 0.005  # 0.003
    
    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = self.read_m(first_item + i)
            if item_id == 0 or item_id == 0xff:
                break
            item_ids.append(item_id + one_indexed)
        return item_ids
    def get_all_item_ids_obs(self):
        # max 85
        return np.array(self.get_items_obs(), dtype=np.uint8)
    
    def _get_obs(self):
        
        screen = self.render()

        self.update_recent_screens(screen)
        
        # normalize to approx 0-1
        sum_of_the_party_level: int =  sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": sum_of_the_party_level,
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "caught_pokemon": self.caught_pokemon,
            "seen_pokemon": self.seen_pokemon,
            "moves_obtained": self.moves_obtained,
        }

        return observation

    def step(self, action):

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()
        
        self.update_pokedex()
        
        self.update_moves_obtained()

        self.party_size = self.read_m(0xD163)

        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        #if step_limit_reached:
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, {}
    '''
    This is the oold run action on emulators function
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                # release button
                self.pyboy.send_input(self.release_actions[action])
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    '''

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels:list[int]  =[
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        # This where the weight and bias stuff is stor
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": self.get_map_location(map_n),
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "party_count": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths_count": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "moves_obtained": int(sum(self.moves_obtained)),
                'visited_pokecenterr': self.progress_reward['visited_pokecenter'],
                'hmr': self.progress_reward['hm'],
                'hm_mover': self.progress_reward['hm_move'],
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad*4, self.coords_pad*4), 
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:,:,0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:,:,0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )
    def get_game_coords(self):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        return x_pos, y_pos, map_n
    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        c = (np.array([x_pos,-y_pos])
        + self.get_map_location(map_n)["coordinates"]
        + self.coords_pad*2)
        return self.explore_map.shape[0]-c[1], c[0]

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0]-self.coords_pad:c[0]+self.coords_pad,
                c[1]-self.coords_pad:c[1]+self.coords_pad
            ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:,:, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:,:, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

        """
        if done:
            self.all_runs.append(self.progress_reward)
            with open(
                self.s_path / Path(f"all_runs_{self.instance_id}.json"), "w"
            ) as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                compression="gzip",
                mode="a",
            )
        """

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end) 
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self)->int:
        '''
        explore_threshould = 22
        scale_factor = 4
        sum_of_the_party_level = self.get_levels_sum()
        if sum_of_the_party_level < explore_threshould:
            scaled =sum_of_the_party_level 
        else:
            scaled = (sum_of_the_party_level - explore_threshould) / scale_factor + explore_threshould
        self.max_level_rew = max(self.max_level_rew, scaled)
        '''
        return self.get_levels_sum()

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def not_boey_get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            self.all_events_string.count('1')
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )
    @property
    def all_events_string(self):
        # cache all events string to improve performance
        if not self._all_events_string:
            event_flags_start = 0xD747
            event_flags_end = 0xD886
            result = ''
            for i in range(event_flags_start, event_flags_end):
                result += bin(self.read_m(i))[2:]  # .zfill(8)
            self._all_events_string = result
        return self._all_events_string
    
    @property
    def battle_type(self):
        if not self._battle_type:
            result = self.read_m(0xD057)
            if result == -1:
                return 0
            return result
        return self._battle_type
    
    def is_wild_battle(self):
        return self.battle_type == 1
    
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
    def init_caches(self):
        # for cached properties
        self._all_events_string = ''
        self._battle_type = -999
    def get_game_state_reward(self, print_stats=True):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        self.max_event_rew = self.update_max_event_rew()
        state_scores = {
            "event":  self.update_max_event_rew(),
            "level":  self.get_levels_reward() * 2,
            "heal":  self.total_healing_rew,
            "op_lvl":  self.update_max_op_level(),
            "dead":  self.died_count ,
            "badge":  self.get_badges() ,
            "explore":  len(self.seen_coords)   * self.explore_weight,
            "seen_pokemon":  sum(self.seen_pokemon) * 2 , 
            "caught_pokemon":  ( sum(self.caught_pokemon) - 1 )  * 2  , 
            "moves_obtained":   ( sum(self.moves_obtained) - 2 ) * 2  , 
            'visited_pokecenter': self.get_visited_pokecenter_reward(),
            'hm': self.get_hm_rewards(),
            'hm_move': self.get_hm_move_reward(),
        }
        # multiply by reward scale
        state_scores = {k: v * self.reward_scale for k, v in state_scores.items()}
        return state_scores
    def update_seen_coords(self):
        ## Getting Game Coordinates
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        self.seen_coords[coord_string] = self.step_count
    def init_map_mem(self):
        self.seen_coords = {}

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level
    
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount
            else:
                self.died_count += 1
    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.pyboy.get_memory_value(i + 0xD2F7)
            seen_mem = self.pyboy.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0

    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.pyboy.get_memory_value(i) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(i + j + 8)
                    if move_id != 0:
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.get_memory_value(0xda80)):
            offset = i*box_struct_length + 0xda96
            if self.pyboy.get_memory_value(offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
    
    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))
    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def get_last_pokecenter_id(self):
        
        last_pokecenter = self.read_m(0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        return self.pokecenter_ids.index(last_pokecenter)
    def get_last_pokecenter_obs(self):
        return self.get_last_pokecenter_list()

    def get_visited_pokecenter_obs(self):
        result = [0] * len(self.pokecenter_ids)
        for i in self.visited_pokecenter_list:
            result[i] = 1
        return result
    def get_visited_pokecenter_reward(self):
        # reward for first time healed in pokecenter
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)
        return len(self.visited_pokecenter_list) * 2
    
    
    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))
    
    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1
    def get_menu_restricted_action(self, action: int) -> int:
        if not self.is_in_battle():
            if self.is_in_start_menu():
                # not in battle and in start menu
                # if wCurrentMenuItem == 1, then up / down will be changed to down
                # if wCurrentMenuItem == 2, then up / down will be changed to up
                current_menu_item = self.read_m(0xCC26)
                if current_menu_item not in [1, 2]:
                    print(f'\nWarning! current start menu item: {current_menu_item}, not 1 or 2')
                    # do nothing, return action
                    return action
                if action < 4:
                    # any arrow key will be changed to down if wCurrentMenuItem == 1
                    # any arrow key will be changed to up if wCurrentMenuItem == 2
                    if current_menu_item == 1:
                        action = 0  # down
                    elif current_menu_item == 2:
                        action = 3  # up
                pass
            else:
                # no in battle and start menu, pressing START
                # opening menu, always set to 1
                self.pyboy.set_memory_value(0xCC2D, 1)  # wBattleAndStartSavedMenuItem
        return action
    def get_hm_rewards(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        total_hm_cnt = 0
        for hm_id in hm_ids:
            if hm_id in items:
                total_hm_cnt += 1
        return total_hm_cnt * 1
    def get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [self.read_m(first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves
    def get_hm_move_obs(self):
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        result = [0] * len(hm_moves)
        all_moves = self.get_party_moves()
        for i, hm_move in enumerate(hm_moves):
            if hm_move in all_moves:
                result[i] = 1
                continue
        return result
    def get_hm_move_reward(self):
        all_moves = self.get_party_moves()
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        hm_move_count = 0
        for hm_move in hm_moves:
            if hm_move in all_moves:
                hm_move_count += 1
        return hm_move_count * 1.5


    def run_action_on_emulator(self, action):
        if self.extra_buttons and self.restricted_start_menu:
            # restrict start menu choices
            action = self.get_menu_restricted_action(action)
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if self.headless and (self.fast_video or not self.save_video):
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def get_map_location(self, map_idx):
        map_locations = {
            0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
            1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
            2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
            3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
            62: {"name": "Invaded house (Cerulean City)", "coordinates": np.array([290, 227])},
            63: {"name": "trade house (Cerulean City)", "coordinates": np.array([290, 212])},
            64: {"name": "Pokémon Center (Cerulean City)", "coordinates": np.array([290, 197])},
            65: {"name": "Pokémon Gym (Cerulean City)", "coordinates": np.array([290, 182])},
            66: {"name": "Bike Shop (Cerulean City)", "coordinates": np.array([290, 167])},
            67: {"name": "Poké Mart (Cerulean City)", "coordinates": np.array([290, 152])},
            35: {"name": "Route 24", "coordinates": np.array([250, 235])},
            36: {"name": "Route 25", "coordinates": np.array([270, 267])},
            12: {"name": "Route 1", "coordinates": np.array([70, 43])},
            13: {"name": "Route 2", "coordinates": np.array([70, 151])},
            14: {"name": "Route 3", "coordinates": np.array([100, 179])},
            15: {"name": "Route 4", "coordinates": np.array([150, 197])},
            33: {"name": "Route 22", "coordinates": np.array([20, 71])},
            37: {"name": "Red house first", "coordinates": np.array([61, 9])},
            38: {"name": "Red house second", "coordinates": np.array([61, 0])},
            39: {"name": "Blues house", "coordinates": np.array([91, 9])},
            40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
            41: {"name": "Pokémon Center (Viridian City)", "coordinates": np.array([100, 54])},
            42: {"name": "Poké Mart (Viridian City)", "coordinates": np.array([100, 62])},
            43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
            44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
            47: {"name": "Gate (Viridian City/Pewter City) (Route 2)", "coordinates": np.array([91,143])},
            49: {"name": "Gate (Route 2)", "coordinates": np.array([91,115])},
            50: {"name": "Gate (Route 2/Viridian Forest) (Route 2)", "coordinates": np.array([91,115])},
            51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
            52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
            53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
            54: {"name": "Pokémon Gym (Pewter City)", "coordinates": np.array([49, 176])},
            55: {"name": "House with disobedient Nidoran♂ (Pewter City)", "coordinates": np.array([51, 184])},
            56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
            57: {"name": "House with two Trainers (Pewter City)", "coordinates": np.array([51, 184])},
            58: {"name": "Pokémon Center (Pewter City)", "coordinates": np.array([45, 161])},
            59: {"name": "Mt. Moon (Route 3 entrance)", "coordinates": np.array([153, 234])},
            60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
            61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
            68: {"name": "Pokémon Center (Route 3)", "coordinates": np.array([135, 197])},
            193: {"name": "Badges check gate (Route 22)", "coordinates": np.array([0, 87])}, # TODO this coord is guessed, needs to be updated
            230: {"name": "Badge Man House (Cerulean City)", "coordinates": np.array([290, 137])}
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return {"name": "Unknown", "coordinates": np.array([80, 0])} # TODO once all maps are added this case won't be needed