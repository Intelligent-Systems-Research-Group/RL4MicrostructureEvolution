import gym
from pathlib import Path
import random
from datetime import datetime
from gym import logger

from msevolution_env.envs.microstructure_evolution import DiscreteMsEvolutionWrapper
from msevolution_env.fem_env import FEMCSVLogger
from msevolution_env.fem_wrapper import ODF

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
    outdir = Path().absolute().joinpath(f"out_meg_random_agt/{datetime.now().strftime('%y%m%d_%H_%M')}")

    env = gym.make('ms-evolution-mggs-v0', storage_path=outdir)
    action_oris = Path().absolute().joinpath('../assets/ud_oris/n100-id1_triclinic.ori')
    env = DiscreteMsEvolutionWrapper(env,
                                     strain_vals=[-1, 1],
                                     strain_rotation_orientations_file=action_oris,
                                     zero_action=True)

    env = FEMCSVLogger(env, outdir=outdir)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    goal_files = [f for f in sorted(Path().absolute().
                                    joinpath('../assets/microstructures/min_equiv_strain_goals').glob('*'))]
    goal_files = [f for f in goal_files if f.suffix == '.dat']
    goal_odfs = [ODF.from_ori_file(f) for f in goal_files]

    for i in range(episode_count):

        ob = env.reset()
        # run episode
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            print("================== example calls for MEG Environment ==================")
            potential_obs = env.get_potential_observation(goal_odfs[0], ob)
            pot_rewards = env.get_potential_rewards(goal_odfs)
            pot_dists = env.get_distances(goal_odfs)
            print(f"pot-rew:\t{pot_rewards}\n pot-dist:\t{pot_dists}")
            print("=======================================================================")

            if done:
                # set random goal
                rnd_i = random.randrange(0, len(goal_odfs))
                print(f"set random goal-texture: {goal_files[rnd_i]}")
                env.set_next_eps_goal(goal_odfs[rnd_i])
                break
    # Close the env and write monitor result info to disk
    env.close()
