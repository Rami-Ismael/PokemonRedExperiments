import argparse
import multiprocessing
import random
import numpy as np
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from rich import print
from utils import exploration_map_callback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    
    set_random_seed(seed)
    
    return _init

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action = "store_true")
    parser.add_argument("--n-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--use-wandb-logging", action="store_true")
    parser.add_argument("--ep-length", type=int, default = 8168)
    parser.add_argument("--sess-id", type=str, default=str(uuid.uuid4())[:8])
    parser.add_argument("--save-video", action='store_true')
    parser.add_argument("--fast-video", action='store_true')
    parser.add_argument("--frame-stacks", type=int, default = 32)
    parser.add_argument("--policy", choices=["MultiInputPolicy", "CnnPolicy"], default="MultiInputPolicy")
    parser.add_argument("--explore-weight", type=float, default = 32)
    parser.add_argument("--reward-scale", type=float, default = 0.05)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--early_stop", action = "store_true")
    parser.add_argument("--extra-buttons", action = "store_true")
    parser.add_argument("--restricted-start-menu", action = "store_true")
    parser.add_argument("--use_screen_explore", action = "store_false")
    parser.add_argument("--randomize-fist-ep-split-cnt", type=int, default = 0)
    parser.add_argument("--similar-frame-dist", type=int, default = 2_000_000.0)
   
    # Arguments 
    args = parser.parse_args()
    print(f"The value of the headless is {args.headless}")

    use_wandb_logging = True
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
        "headless": args.headless,
        "save_final_state": True,
        "early_stop": args.early_stop,
        "action_freq": 24,
        "init_state": "has_pokedex_nballs.state",
        "max_steps": args.ep_length,
        "print_rewards": True,
        "save_video": args.save_video,
        "fast_video": args.fast_video,
        "session_path": sess_path,
        "gb_path": 'PokemonRed.gb',
        "debug": False,
        "sim_frame_dist": args.similar_frame_dist,
        "use_screen_explore": True,
        "reward_scale": args.reward_scale,
        "extra_buttons": False,
        "explore_weight": args.explore_weight,  # 2.5
        "explore_npc_weight": 1,  # 2.5
        "frame_stacks": args.frame_stacks,
        "policy": args.policy,
        "restricted_start_menu": args.restricted_start_menu,
        "extra_buttons": args.extra_buttons,
        "use_screen_explore": args.use_screen_explore,
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"The environment config is: {env_config}")
    import os
    num_cpu = os.cpu_count() //2   # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=args.ep_length, 
                                             save_path=sess_path,
                                            name_prefix="poke")

    callbacks = [checkpoint_callback, TensorboardCallback(sess_path), exploration_map_callback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            #name="less-event-log-text-test-logs-stack3-all-obs",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    # put a checkpoint here you want to start from
    file_name = "" #"session_9ff8e5f0/poke_21626880_steps"
    
    
    
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps =args.ep_length 
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = args.ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO( args.policy, 
                    env, verbose=1, 
                    n_steps = args.ep_length,
                    batch_size = 128, 
                    n_epochs = 3    , 
                    gamma=0.998, 
                    tensorboard_log=sess_path , 
                    seed = args.seed,
                    ent_coef = 0.01,
        )

    print(model.policy)

    model.learn(total_timesteps=(args.ep_length)*num_cpu * 40, 
                callback=CallbackList(callbacks), 
                progress_bar=True, 
                log_interval = 2)

    if use_wandb_logging:
        run.finish()
