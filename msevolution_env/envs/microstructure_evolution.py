import logging
import numpy as np
from gym import spaces, Wrapper
from pathlib import Path

from msevolution_env.fem_env import FEMEnv
from msevolution_env.fem_wrapper import MStructEvoWrapper, ODF


class MSEvolution(FEMEnv):
    ENV_ID = 'ms-evolution-v1'

    action_names = ['d', 'Q1', 'Q2', 'Q3', 'Q4']
    action_values = None

    # stable-baselines recommends symmetric and normalized Box action space (range=[-1, 1])
    # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    """
    action_space = spaces.Box(low=np.array([-0.02, -1.0, -1.0, -1.0, -1.0]),
                              high=np.array([0.02, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
    """

    fem_engine = "MStructEvoV2"
    # TIME_STEPS = 25

    # gsh parameters
    # CRYSTAL_SYM = 'cubic'
    # SAMPLE_SYM = 'triclinic'
    # GSH_DEGREE = 12

    # hist parameters
    # NEIGHBOR_COUNT = 3

    _curr_dist = None
    _SIM_ERROR_REWARD = 0
    _standard_img_size = (1440, 1243, 3)
    _standard_img = np.zeros(_standard_img_size, dtype=np.uint8)
    np.set_printoptions(precision=4)

    def __init__(self, storage_path, gamma=1.0):
        super().__init__(storage_path)

        # read in configuration
        self.MAX_TIME_STEPS = 100
        self._distance_measure = 'whd_chi_square'
        self.add_timestep = True
        self.add_eps_eq = True
        self.add_dist = True
        obs_count = self._get_obs_count()
        self.observation_space = spaces.Box(-1, 1, shape=[obs_count], dtype='float32')

        self._strain_bounds = [-0.02, 0.02]
        assert len(self._strain_bounds) == 2
        assert self._strain_bounds[0] <= self._strain_bounds[1]

        # set templates (shallow structure)
        self._simulation_id_templates = []
        for i in range(self.MAX_TIME_STEPS):
            s = 'ts_{ts}'
            self._simulation_id_templates.append(s)

        self._fem_wrapper = MStructEvoWrapper(self.sim_storage,
                                              material='250ud_oris.inp',
                                              cpu_kernels=40,
                                              terminal_state_plot_folder=Path(self.sim_storage).joinpath(
                                                  'terminal_state_visualizations'))

        goal_file = Path(__file__).parent.joinpath('../assets/microstructures/min_equiv_strain_goals/g0.dat')
        self._goal_odf = ODF.from_ori_file(goal_file)
        self._goal_dist_tolerance = 0.1  # todo check for scale

        self._curr_microstructure = self._fem_wrapper.read_simulation_results('initial')[0]

        self.success = False

        self._best_ts = None
        self._episode_best_dist = None

        # todo let the agent do the shaping
        self.gamma = gamma

    def _get_obs_count(self):
        obs_count = 42  # todo assuming gsh degree 8
        for additional_scalar in [self.add_timestep, self.add_dist, self.add_eps_eq]:
            if additional_scalar:
                obs_count += 1
        return obs_count

    def _scale_strain(self, strain):
        # scale
        scaled_strain = ((strain + 1) / 2) * (self._strain_bounds[1] - self._strain_bounds[0]) + self._strain_bounds[0]
        # avoid simulation instabilities by disallowing mini-strain
        if abs(scaled_strain) < 1e-4:
            scaled_strain = -1e-4 if scaled_strain < 0 else 1e-4
        return scaled_strain

    def step(self, action):
        action[0] = self._scale_strain(action[0])
        print(f'ts: {self.time_step}, eps: {self.episode} action: {action}')
        logging.debug(f'step ts: {self.time_step} action: {action}')
        o, r, done, info = super().step(action)
        if self.simulation_failed:
            # simulation error
            # unchanged dist
            self._curr_dist = self._get_distance_to_goal_odf(self._curr_microstructure)
            self._curr_microstructure = self._fem_wrapper.read_simulation_results('initial')[0]
        else:
            self._curr_microstructure = self._curr_fem_results[0]
            self._curr_dist = self._get_distance_to_goal_odf(self._curr_microstructure)

        self.success = self._goal_reached(self._curr_microstructure)
        info['is_success'] = self.success
        return o, r, done, info

    def reset(self):
        super().reset()
        # load ms from mat.inp
        initial_results = self._fem_wrapper.read_simulation_results('initial')
        self._curr_microstructure = initial_results[0]
        # calc curr_dist
        self._curr_dist = self._get_distance_to_goal_odf(initial_results[0])
        self._best_ts = 0
        self._episode_best_dist = self._curr_dist

        return self._apply_observation_function(initial_results)

    def _apply_observation_function(self, fem_results):
        if fem_results is None:
            o = np.zeros(42)  # todo assuming degree 8
        else:
            odf = fem_results[0]
            o = odf.get_gsh()

        if self.add_timestep:
            o = np.append(o, (self.time_step + 1) / self.MAX_TIME_STEPS)
        if self.add_eps_eq:
            if self.time_step == 0:
                eps_eq = 0.0
            else:
                eps_eq = self._fem_wrapper.get_eps_eq()
            assert eps_eq < 1.0
            o = np.append(o, eps_eq)
        if self.add_dist:
            o = np.append(o, min(self._get_distance_to_goal_odf(self._curr_microstructure) / 100, 1))
        return o

    def _apply_reward_function(self, fem_results):
        d = self._get_distance_to_goal_odf(fem_results[0])
        self.info['dist'] = d

        r = self._calc_dist_based_reward(d, self._curr_dist)
        if d < self._episode_best_dist:
            self._episode_best_dist = d
            self._best_ts = self.time_step
        self.info['episode_best_dist_ts'] = self._best_ts
        self.info['episode_best_dist'] = self._episode_best_dist

        print(f'distance to goal-microstructure: {d}, previous: {self._curr_dist}, R: {r}')
        return r

    def _calc_dist_based_reward(self, d, prev_d=None):
        # following https://dl.acm.org/doi/10.5555/645528.657613
        r = (1 / d) if self._is_done() else 0.0
        if self._is_done():
            f = 0 - (1 / prev_d)
        else:
            f = self.gamma * (1 / d) - (1 / prev_d)

        return r + f

    def _calc_sim_error_reward(self):
        return 0

    def _render_state(self):
        return self._fem_wrapper.get_state_visualization(time_step=self.time_step,
                                                         odf=self._curr_microstructure,
                                                         goal_odf=self._goal_odf,
                                                         distance_measure=self._distance_measure)

    def _get_distance_to_goal_odf(self, odf):
        return odf.get_distance(self._goal_odf, self._distance_measure)

    def _is_done(self, state=None):
        # if either max-time-steps or distance within a given range
        if self.simulation_failed:
            return True
        if self.time_step == self.MAX_TIME_STEPS - 1:
            return True
        if self._goal_reached(state):
            return True
        return False

    def _goal_reached(self, state=None):
        if state is None:
            if self._curr_microstructure is None:
                return False
            else:
                return self._get_distance_to_goal_odf(self._curr_microstructure) < self._goal_dist_tolerance
        else:
            return self._get_distance_to_goal_odf(state) < self._goal_dist_tolerance

    def _get_state_vector(self, fem_results):
        # decode quaternions and create np array
        odf = fem_results[0]
        return np.array([q.elements for q in odf.get_oris()]).flatten()

    def _sample_process_conditions(self):
        return {}

    def set_goal(self, goal_odf):
        self._goal_odf = goal_odf
        self._curr_dist = self._get_distance_to_goal_odf(self._curr_microstructure)

    def get_current_goal(self):
        return self._goal_odf


class MSEvolutionMultiGoal(MSEvolution):
    ENV_ID = 'ms-evolution-mg-v1'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_microstructure = None

    def reset(self):
        self._last_microstructure = None
        return super().reset()

    def step(self, action):
        self._last_microstructure = self._curr_microstructure
        return super().step(action)

    def get_distances(self, goals, microstructure=None):
        if self.simulation_failed:
            microstructure = self._last_microstructure
        if microstructure is None:
            microstructure = self._curr_microstructure
        dists = np.array(
            [microstructure.get_distance(goal, self._distance_measure) for goal in goals])
        return dists

    def get_potential_rewards(self, goals):
        if self.simulation_failed:
            return np.ones(len(goals)) * self._SIM_ERROR_REWARD

        curr_dists = self.get_distances(goals)
        prev_dists = self.get_distances(goals, self._last_microstructure)
        return np.array([self._calc_dist_based_reward(d, prev_d) for d, prev_d in zip(curr_dists, prev_dists)])


class MSEvolutionMultiGoalWGoalStates(MSEvolutionMultiGoal):
    ENV_ID = 'ms-evolution-mggs-v0'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        n_additional_scalars = [self.add_dist, self.add_eps_eq, self.add_timestep].count(True)
        self.observation_space = spaces.Box(-1, 1, shape=[self.observation_space.shape[0] * 2 - n_additional_scalars],
                                            dtype='float32')

        self._next_eps_goal_odf = None
        self._goal_reached_last_episode = False

    def step(self, action):
        o, r, done, info = super().step(action)
        info['episode_goal'] = hash(self._goal_odf)
        return self._get_goal_observation(o), r, done, info

    def reset(self):
        self._goal_reached_last_episode = self._goal_reached()
        if self._next_eps_goal_odf is not None:
            self._goal_odf = self._next_eps_goal_odf
            self._next_eps_goal_odf = None
        o = super().reset()
        return self._get_goal_observation(o)

    def set_next_eps_goal(self, goal_odf):
        self._next_eps_goal_odf = goal_odf

    def get_potential_observation(self, goal_odf, original_observation):
        obs_count = self._get_obs_count()

        observation = original_observation[:obs_count]
        return self._get_goal_observation(observation, goal_odf)

    def _get_goal_observation(self, observation, goal_odf=None):
        if goal_odf is None:
            goal_odf = self._goal_odf

        g = goal_odf.get_gsh()

        return np.concatenate([observation, g])


class DiscreteMsEvolutionWrapper(Wrapper):
    def __init__(self, env, strain_vals, strain_rotation_orientations_file, zero_action=False):
        self._strain_vals = strain_vals
        self._rotations_q = np.loadtxt(strain_rotation_orientations_file, ndmin=2)
        super().__init__(env)
        self.action_count = len(self._strain_vals) * len(self._rotations_q)
        if zero_action:
            self.action_count += 1

        self.action_space = spaces.Discrete(self.action_count)

    def step(self, action):
        print(f'a: {action}')
        if action == 0:
            a = np.hstack([0, self._rotations_q[0]])
        else:
            action -= 1
            a = np.hstack([self._strain_vals[action // len(self._rotations_q)],
                           self._rotations_q[action % len(self._rotations_q)]])
        return super().step(a)
