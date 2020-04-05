import gym
import numpy as np


class normallized_action_wrapper(gym.ActionWrapper):
    # * because the tanh value range is [-1, 1], so change the env action range
    def action(self, action):
        # * change action range from [-1, 1] to [env.low, env.high]
        low = self.action_space.low
        high = self.action_space.high

        action = (action + 1) / 2 * (high - low) - 2
        action = np.clip(action, low, high)
        return action

    def reverse_action(self):
        # * change action range from [env.low, env.high] to [-1, 1]
        low = self.action_space.low
        high = self.action_space.high

        action = (action - low) / ((high - low) / 2) - 1
        action = np.clip(action, -1, 1)
        return action
