# based on https://github.com/openai/baselines/blob/master/baslines/common/atari_wrappers.py
import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
#from matplotlib import pyplot

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class TwoLatestFramesEnv(gym.Wrapper):
    def __init__(self, env):
        """Return the last two observations"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque([], maxlen=2) #np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self.idx = 0

    def step(self, action):
        """Store the last observation and return the storage. 
        Note that the observation is mutable and must be processed by np.maximum() before using or storing 
        """
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs) 
        return self._obs_buffer, reward, done, info

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(2):
            self._obs_buffer.append(ob)
        return self._obs_buffer


class MaxAndSkipAndRewardClipEnv(gym.Wrapper):
    def __init__(self, env, skip=4, clip=True):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert type(skip) == int and skip > 0
        self._skip       = skip
        self.clip        = clip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        total_clipped_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if self.clip:
                total_clipped_reward += np.sign(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = np.maximum(*obs, dtype=np.uint8)
        reward = (total_reward, total_clipped_reward) if self.clip else total_reward
        return max_frame, reward, done, info

    def reset(self, **kwargs):
        return np.maximum(*self.env.reset(**kwargs), dtype=np.uint8)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0 # env.reset() will be called before it starts. Therefore this initial value does not matter.
        self.was_real_done  = True
        # If it is not waiting for reset, we regard the reset as an accident and we truly reset the environment.
        self.waiting_for_reset = False
        self.reset_ob = None
    def step(self, action):
        assert not self.waiting_for_reset
        ob, reward, done, info = self.env.step(action)
        self.was_real_done = done # this includes the case of "TimeLimit.truncated"
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        # usually "done" must be False with "lives < self.lives and lives > 0", but with "TimeLimit" it can be True
        if lives < self.lives and lives > 0 and not done: 
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            # no-op step to advance from terminal/lost life state
            # In order to avoid problems caused by the step "self.env.step(0)" during the reset, we execute "self.env.step(0)" in advance.
            self.reset_ob, _, self.was_real_done, _ = self.env.step(0)
            self.waiting_for_reset = True
        self.lives = lives
        return ob, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if not self.was_real_done and self.waiting_for_reset: 
            # extract the result of the previous no-op step
            ob = self.reset_ob 
            self.reset_ob = None
        else: 
            ob = self.env.reset(**kwargs)
            self.was_real_done = False

        self.lives = self.env.unwrapped.ale.lives()
        self.waiting_for_reset = False
        return ob


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env, downscale=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        super().__init__(env)
        # if the size of the image is 160 x 250, we crop it from the top by 28 and from the bottom by 12, so that it becomes the default 160 x 210
        shp = env.observation_space.shape
        self.crop40 = True if shp[0] == 250 else False
        if downscale > 10:
            self._width = downscale
            self._height = downscale
        else:
            assert downscale > 0, "invalid downscaling ratio {}".format(downscale) 
            self._width = round(shp[1]/downscale) 
            self._height = round(210/downscale) if self.crop40 else round(shp[0]/downscale) 

        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3
        # we move the color channel dimension from shape[-1] to shape[0] 
        new_space = gym.spaces.Box(low=0, high=255,
            shape=(num_colors, self._height, self._width), dtype=np.uint8)

        self.observation_space = new_space 

    def observation(self, obs):
        frame = obs
        if self.crop40: frame = frame[28:238,:,:] # crop from the top by 28 and from the bottom by 12
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # numpy array treated as an image has a shape of (height, width, color)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        frame = np.ascontiguousarray(np.moveaxis(frame, -1, 0))

        return frame

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.k = k
        self.frames = deque([], maxlen=k)
        # the color channel dimension has been moved from shape[-1] to shape[0] in "WrapFrame"
        # the observation is supposed to be accessed via np.array(...)
        self.observation_space = gym.spaces.Box(low=np.float32(0.0), high=np.float32(1.0), shape=((shp[0] * k,) + tuple(shp[1:])), dtype=np.float32)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(tuple(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        # We avoid using the following self._out so that no float32 data will be saved into memory 
        #self._out = None 

    def _force(self): 
        # the data type in self._frames is np.uint8 
        return np.concatenate(self._frames, axis=0).astype(np.float32)/255. 

    def __array__(self):
        return self._force()

    def __len__(self):
        return len(self._force())

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1 
        if self._elapsed_steps > self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return ob, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

def make_atari(env_id, max_episode_steps=None, clip_reward = True):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = TwoLatestFramesEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipAndRewardClipEnv(env, skip=4, clip=clip_reward)
    # the number of steps measured by "TimeLimit" is after the step skipping of "MaxAndSkipEnv", and therefore is 1/4 of real frames
    if max_episode_steps is not None and max_episode_steps>0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=True, frame_stack=4, downscale=84, greyscale=True):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        # 'FIRE' is often needed to restart the game when it loses a life
        env = FireResetEnv(env)
    env = WrapFrame(env, downscale=downscale, grayscale=greyscale)
    frame_stack = int(frame_stack)
    frame_stack = max(frame_stack, 1)
    env = FrameStack(env, frame_stack)
    return env

def wrap_atari_dqn(env, args):
    env = wrap_deepmind(env, 
                        episode_life=args.episode_life, 
                        frame_stack=args.frame_stack, 
                        downscale=args.frame_downscale,
                        greyscale=args.grey) 
    return env
