import sys
sys.path.append('../habitat-av')
from habitat_audio import *

import itertools
from collections import deque
from collections import defaultdict
from copy import copy
import scipy.io.wavfile as wv

import gym
import numpy as np
from PIL import Image


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        audio = []
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward

            if 'audio' in info:
                # Keep audio from skipped frames
                audio.extend(info['audio'])
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        if len(audio) > 0:
            acc_info['audio'] = np.asarray(audio)
        return max_frame, total_reward, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # atari resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        elif frame.size == 128 * 128 * 3:  # habitat resolution
            img = np.reshape(frame, [128, 128, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(
            size, resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        audio = []
        for _ in range(self.n):
            #self.env.render()
            ob, rew, done, info = self.env.step(action)
            totrew += rew

            # Keep audio from skipped frames
            audio.extend(info['audio'])
            if done:
                break

        info['audio'] = np.asarray(audio)
        return ob, totrew, done, info


def make_habitat():
    from habitat.datasets import make_dataset
    from habitat_baselines.common.environments import NavRLEnv
    from default import get_config
    from baselines.common.atari_wrappers import FrameStack
    
    config = get_config(config_paths="/home/vdean/habitat-av/configs/tasks/pointnav_nomap.yaml")
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
    config.freeze()
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE,
                           config=config.TASK_CONFIG.DATASET)
    env = NavRLEnv(config=config, dataset=dataset)
    env = HabitatWrapper(env)
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)
    return env

class HabitatWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.state_counts = defaultdict(int)
        self.state_counts_recent = defaultdict(int)
        self.states_visited = {}
        self.states_visited_recent = {}

    def reset(self):
        print("Total visited states:",self.states_visited.keys())
        print("Recently visited states:",self.states_visited_recent.keys())
        print("Total visited state counts:",self.state_counts)
        print("Recently visited state counts:",self.state_counts_recent)
        self.states_visited_recent = {}
        self.state_counts_recent = defaultdict(int)
        return np.asarray(self.env.reset()['rgb'])

    def step(self, action):
        # Action 0 is reset, so just use actions 1 (move forward), 2 (turn left), and 3 (turn right)
        ob, rew, done, info = self.env.step(**{'action':action+1})
        position = self.env._env._sim.get_agent_state().position
        xy_position = tuple([position[0],position[2]])
        self.states_visited[xy_position] = True
        self.states_visited_recent[xy_position] = True
        self.state_counts[xy_position] += 1
        self.state_counts_recent[xy_position] += 1
        info['n_states_visited'] = len(self.states_visited.keys())
        info['n_states_visited_recent'] = len(self.states_visited_recent.keys())
        if info['audio_file'] is not None:
            rate, info['audio'] = wv.read(info['audio_file'])
        return np.asarray(ob['rgb']), rew, done, info
