from vime import vime
from sac import sac_vime
import gym
from env_wrapper import normallized_action_wrapper

if __name__ == '__main__':
    env = normallized_action_wrapper(gym.make('Pendulum-v0'))
    vime_model = vime(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_size=64,
        min_logvar=-10,
        max_logvar=2,
        learning_rate=1e-4,
        kl_buffer_capacity=10,
        lamda=1e-2,
        update_iteration=500,
        batch_size=10,
        eta=1e-1
    )
    test = sac_vime(
        env=env,
        batch_size=100,
        learning_rate=1e-3,
        exploration=1,
        episode=100000,
        gamma=0.99,
        alpha=0.2,
        capacity=100000,
        rho=0.995,
        update_iter=10,
        update_every=50,
        render=False,
        log=False,
        vime_model=vime_model
    )
    test.run()