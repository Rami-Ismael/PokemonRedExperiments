import numpy as np
def make_pokemon_red_overlay(bg, counts):
    nonzero = np.where(counts > 0, 1, 0)
    scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.zeros((*counts.shape, 3))
    hsv[..., 0] =  (240.0 / 360) - scaled * (240.0 / 360.0) # heatmap, not coldmap. scaled*(240.0/360.0)
    hsv[..., 1] = nonzero
    hsv[..., 2] = nonzero

    # Convert the HSV image to RGB
    import matplotlib.colors as mcolors
    overlay = 255*mcolors.hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ones((16, 16, 1), dtype=np.uint8)
    overlay = np.kron(overlay, kernel).astype(np.uint8)
    mask = np.kron(nonzero, kernel[..., 0]).astype(np.uint8)
    mask = np.stack([mask, mask, mask], axis=-1).astype(bool)

    # Combine with background
    render = bg.copy().astype(np.int32)
    render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render
from stable_baselines3.common.callbacks import BaseCallback


class exploration_map_callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        try:
            for k, v in self.locals.items():
                if 'Task_eval_fn' in k:
                    # Temporary hack for NMMO competition
                    continue
                if 'pokemon_exploration_map' in k:
                    import cv2
                    bg = cv2.imread('kanto_map_dsv.png')
                    overlay = make_pokemon_red_overlay(bg, sum(v))
                    if self.model.wandb is not None:
                        self.model.wandb.log({f"Media/exploration_map": self.model.wandb.Image(overlay)})
                try: # TODO: Better checks on log data types
                    self.logger.record(f'callback/{k}', np.mean(v))
                except:
                    continue
        except Exception as e:
            print(e)
        pass
"""
    data.stats = {}
    for k, v in infos['learner'].items():
        if 'Task_eval_fn' in k:
            # Temporary hack for NMMO competition
            continue
        if 'pokemon_exploration_map' in k:
            import cv2
            bg = cv2.imread('kanto_map_dsv.png')
            overlay = make_pokemon_red_overlay(bg, sum(v))
            if data.wandb is not None:
                data.stats['Media/exploration_map'] = data.wandb.Image(overlay)
        try: # TODO: Better checks on log data types
            data.stats[k] = np.mean(v)
        except:
            continue

    if config.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    return data.stats, infos
"""