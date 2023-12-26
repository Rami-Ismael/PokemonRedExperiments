import argparse
import multiprocessing
import os
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rom-path", default="../PokemonRed.gb")
    parser.add_argument("--state-path", default="../home.state")
    parser.add_argument("--n-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--use-wandb-logging", action="store_true")
    parser.add_argument("--ep-length", type=int, default=2048 * 10)
    parser.add_argument("--sess-id", type=str, default=str(uuid.uuid4())[:8])
    parser.add_argument("--save-video", action='store_true')
    parser.add_argument("--fast-video", action='store_true')

    use_wandb_logging = True
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    
    
    
    
    
    sess_path = Path(f'session_{sess_id}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': True,
                'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    num_cpu = os.cpu_count() // 2  # Also sets the number of episodes per training iteration
    print(f'Using {num_cpu} CPUs')
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    
    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)
    learn_steps = 40
    print(f'Learning for {learn_steps} steps')
    # put a checkpoint here you want to start from
    file_name = 'session_e41c9eff/poke_38207488_steps' 
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('\ncreating new model')
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)
        print(f"The model policy is {model.policy}")
    
    print(f"The total timesteps is {ep_length*num_cpu*1000}")
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
