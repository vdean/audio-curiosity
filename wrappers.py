from collections import deque

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
        elif frame.size == 224 * 240 * 3:
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
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


class RetroALEActions(gym.ActionWrapper):
    def __init__(self, env, all_buttons, n_players=1):
        gym.ActionWrapper.__init__(self, env)
        self.n_players = n_players
        self._num_buttons = len(all_buttons)
        bs = [-1, 0, 4, 5, 6, 7]

        def update_actions(old_actions, offset=0):
            actions = []
            for b in old_actions:
                for button in bs:
                    action = []
                    action.extend(b)
                    if button != -1:
                        action.append(button + offset)
                    actions.append(action)
            return actions

        current_actions = [[]]
        for i in range(self.n_players):
            current_actions = update_actions(current_actions, i * self._num_buttons)
        self._actions = current_actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons * self.n_players)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.5):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = [0]*8

    def reset(self):
        self.last_action = [0]*8
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


def make_retro(env_name="Breakout", naudio_samples=None,
               sticky_env=False, make_video=False, is_baseline=False):
    import retro
    from baselines.common.atari_wrappers import FrameStack
    env = retro.make(env_name + '-Atari2600', naudio_samples=naudio_samples,
                     make_video=make_video, is_baseline=is_baseline)
    max_episode_steps = 4500
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)
    if not sticky_env:
        env = ExtraTimeLimit(env, max_episode_steps)
    if sticky_env:
        env._max_episode_steps = max_episode_steps * 4
        env = StickyActionEnv(env)
    env = RetroALEActions(env, env.buttons)
    return env
