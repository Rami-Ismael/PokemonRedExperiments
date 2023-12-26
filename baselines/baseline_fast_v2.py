import argparse
import multiprocessing
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
    parser.add_argument("--headless", type=bool, default=True)
    parser.add_argument("--n-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--use-wandb-logging", action="store_true")
    parser.add_argument("--ep-length", type=int, default=2048 * 10)
    parser.add_argument("--sess-id", type=str, default=str(uuid.uuid4())[:8])
    parser.add_argument("--save-video", action='store_true')
    parser.add_argument("--fast-video", action='store_true')
    parser.add_argument("--frame-stacks", type=int, default=4)
    parser.add_argument("--policy", choices=["MultiInputPolicy", "CnnPolicy"], default="MultiInputPolicy")
    parser.add_argument("--explore-weight", type=float, default=5)
   
    # Arguments 
    args = parser.parse_args()

    use_wandb_logging = True
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
        "headless": args.headless,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "has_pokedex_nballs.state",
        "max_steps": args.ep_length,
        "print_rewards": True,
        "save_video": args.save_video,
        "fast_video": args.fast_video,
        "session_path": sess_path,
        "gb_path": 'PokemonRed.gb',
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": False,
        "explore_weight": args.explore_weight,  # 2.5
        "explore_npc_weight": 1,  # 2.5
        "frame_stacks": args.frame_stacks,
        "policy": args.policy,
    }

    print(f"The environment config is: {env_config}")
    import os
    num_cpu = os.cpu_count() // 2  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=args.ep_length, 
                                             save_path=sess_path,
                                            name_prefix="poke")

    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

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
    
    n_steps = args.ep_length // 8
    print(f"Learning for {n_steps} steps")
    
    
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = n_steps
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = args.ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO( args.policy, 
                    env, verbose=1, 
                    n_steps = n_steps , 
                    batch_size=128, 
                    n_epochs = 3    , 
                    gamma=0.998, 
                    tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=(args.ep_length)*num_cpu*10000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()