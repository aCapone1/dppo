"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gym
from gym import spaces
import imageio
import os
import re
import invertedai as iai
from typing import Optional
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecEnv

api_key = open('api_key.txt','r').read()
os.environ['IAI_API_KEY'] = api_key
iai.add_apikey(api_key)


class TorchdriveenvWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta,
        init_state: np.ndarray=None,
        normalization_path: str =None,
        video_name: Optional[str] = None,
    ):
        
        self.env = env

        # setup spaces
        self.video_writer = None
        self.video_path = None
        self.video_num = 0

        normalization = np.load(normalization_path)
        self.init_state = init_state
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]
        self.normalize = normalization_path is not None

        self.observation_space = spaces.Dict()
        # obs_example = self.env.reset()
        _ = self.env.reset()
        if isinstance(self.env, VecEnv):
            obs_example = self.env.unwrapped.envs[0].unwrapped.simulator.get_state().cpu().numpy()
        else:
            obs_example = self.env.unwrapped.simulator.get_state().cpu().numpy()

        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        # for key, value in shape_meta["obs"].items():
        #     shape = value["shape"]
        self.observation_space["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=shape_meta["obs"]["rgb"]["shape"],
                dtype=np.float32,
            )
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        action_space = env.action_space
        self.action_space = spaces.Box(
            low=action_space.low,
            high=action_space.high,
            shape=action_space.shape,
            dtype=action_space.dtype,
        )
        # if not video_name is None:
        #     # render example scenario
        #     self.env = DummyVecEnv([env])
        #     self.env = VecVideoRecorder(
        #         venv=env,
        #         video_folder="log/videos/",
        #         record_video_trigger=lambda x: x % 400 == 0,
        #         video_length=400,
        #         name_prefix=video_name,
        #     )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        # if self.clamp_obs:
        #     obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        obs["rgb"] = raw_obs
        if isinstance(self.env, VecEnv):
            obs["state"] = self.env.unwrapped.envs[0].unwrapped.simulator.get_state().cpu().numpy()
        else:
            obs["state"] = self.env.unwrapped.simulator.get_state().cpu().numpy()

        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_num += 1
            cur_video_path = self.video_path
            self.video_path = re.sub(r"eval_trial-(.*).mp4", r"eval_trial-"+str(self.video_num)+".mp4", cur_video_path)
            self.video_writer = imageio.get_writer(self.video_path)
            # self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_path = options["video_path"]
            self.video_writer = imageio.get_writer(self.video_path)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()[0]
        return self.get_observation(raw_obs)

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, truncated, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

                # render if specified
        if self.video_writer is not None:
            video_img = np.uint8(self.env.render())
            self.video_writer.append_data(video_img)

        return obs, reward, done, info

    # def render(self, mode="rgb_array"):
    #     h, w = self.render_hw
    #     return self.env.render()
