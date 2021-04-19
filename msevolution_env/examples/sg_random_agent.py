import gym
import pathlib
from datetime import datetime
from gym import logger

from msevolution_env.envs.microstructure_evolution import DiscreteMsEvolutionWrapper
from msevolution_env.fem_env import FEMCSVLogger


# run random agent on single-goal environment, used during the experiments from
# Structure-Guided Processing Path Optimization with Deep Reinforcement Learning

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)
    outdir = pathlib.Path().absolute().joinpath(f"out_sg_random_agt/{datetime.now().strftime('%y%m%d_%H_%M')}")

    env = gym.make('ms-evolution-v1', storage_path=outdir)
    action_oris = pathlib.Path().absolute().joinpath('../assets/ud_oris/n100-id1_triclinic.ori')
    env = DiscreteMsEvolutionWrapper(env,
                                     strain_vals=[-1, 1],
                                     strain_rotation_orientations_file=action_oris,
                                     zero_action=True)

    env = FEMCSVLogger(env, outdir=outdir)
    agent = RandomAgent(env.action_space)

    episode_count = 10000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    # Close the env and write monitor result info to disk
    env.close()
