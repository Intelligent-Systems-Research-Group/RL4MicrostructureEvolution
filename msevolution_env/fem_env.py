import gym
import imageio
import inspect
import logging
import numpy as np
import time
import warnings
from abc import ABC
from collections.abc import Iterable
from datetime import datetime
from gym.core import Wrapper
from gym.utils import seeding
from pathlib import Path

from msevolution_env.fem_wrapper import SimulationError
from msevolution_env.helpers import CSVLogger


class FEMEnv(gym.Env, ABC):
    """ abstract class for fem-based environments
    unset class variables have to be set by specific environment
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    # environment-id: used as GYM-environment name and for template-paths and simulation-storage-paths
    ENV_ID = None

    # readable names for individual actions, used for solver-templates
    action_names = []

    # Gym Spaces
    action_space = None
    observation_space = None

    # per time-step string templates for the simulation-id (uses self.simulation_parameters)
    _simulation_id_templates = None

    # FEM-engine used (e.g. "Abaq")
    fem_engine = None
    _fem_wrapper = None

    # reward given if the simulation is not solvable
    _SIM_ERROR_REWARD = 0

    # img used for rgb-, and human rendering when no image is available
    _standard_img = None
    _standard_img_size = None

    def __init__(self, storage_path):
        # current episode
        self.episode = 0
        # current time-step
        self.time_step = 0
        self.simulation_failed = False

        # data returned by step(self, action) for 'FEMLogger' or for special purposes like multiobjective-Learning
        # label prefix conventions to enable generic visualization / agents etc.:
        # rt: reward-term
        # ao: actual observation (without artificial additive noise)
        self.info = {}

        # parameters filled into the solver-template before simulation and used to set the simulation-id
        self.simulation_parameters = {}

        self._curr_fem_results = None
        # used for restart-simulations
        self._root_simulation_id = None
        self._base_simulation_id = None

        # stochastic environment dynamic has to be derived from this
        self._np_random_state = None
        self.seed()

        self.viewer = None
        self.curr_img_path = None
        self.img_ts = -1

        self.persistent_simulation = False
        self.visualize = False
        self.reuse_results = False

        store_state_data = True
        storage = storage_path

        # determine paths for current environment
        if self.persistent_simulation:
            if storage in [None, '', 'None']:
                self.sim_storage = Path(__file__).parent.joinpath(f'sim_storage/{self.ENV_ID}')
            else:
                self.sim_storage = Path(f'{storage}/{self.ENV_ID}')
        else:
            t_string = datetime.now().strftime('%y%m%d_%H_%M')
            self.sim_storage = Path(f'{storage}/{self.ENV_ID}_{t_string}')
            logging.info(f'persistence off, creating temporary simulation-storage: {self.sim_storage}')
        self.sim_storage.mkdir(exist_ok=True, parents=True)

        self._state_log = None
        if store_state_data:
            self._state_log = Path(f'{self.sim_storage}/raw_state_reward_log.log')

    def step(self, action):
        """
        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) :
                amount of reward returned after previous action
            done (boolean):
                whether the episode has ended, in which case further step() calls will return undefined results
            info (dict):
                contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # update simulation-parameters by the current action (supports multiple-input control cases)
        if isinstance(action, Iterable):
            for i, a in enumerate(action):
                self.simulation_parameters[f'{self.action_names[i]}_{self.time_step}'] = a
        else:
            self.simulation_parameters[f'{self.action_names[0]}_{self.time_step}'] = action

        # read out current process conditions
        process_conditions = self._sample_process_conditions()
        self.simulation_parameters.update(process_conditions)
        self.info.update(process_conditions)

        # create simulation-id (for file- and folder-names) from simulation-parameters
        format_dict = self.simulation_parameters.copy()
        format_dict['ts'] = datetime.now().strftime('%m-%d_%H-%M-%S-%f')
        simulation_id = self._simulation_id_templates[self.time_step].format(**format_dict)
        self.info['sim_id'] = simulation_id
        logging.debug(simulation_id)
        i = 1

        # wait while simulation is locked
        while not self._fem_wrapper.request_lock(simulation_id):
            logging.warning(f'waiting for lock release {simulation_id}')
            time.sleep(2 ** i)
            i += 1

        self.simulation_failed = False
        try:
            # check simulation store for results, if none available: simulate
            if self._fem_wrapper.simulation_results_available(simulation_id) and self.reuse_results:
                pass
            else:
                # run simulation
                self._fem_wrapper.run_simulation(simulation_id,
                                                 self.simulation_parameters,
                                                 self.time_step,
                                                 self._base_simulation_id)
        except SimulationError:
            self.simulation_failed = True
            o = self._apply_observation_function(None)
            r = self._calc_sim_error_reward()
            logging.warning(f'{simulation_id} not solvable! Throwing Reward {r}')
            self._curr_fem_results = None
            done = True
        else:
            # read FEM-results
            fem_results = self._fem_wrapper.read_simulation_results(simulation_id,
                                                                    root_simulation_id=self._root_simulation_id)
            self._curr_fem_results = fem_results
            # apply reward function
            r = self._apply_reward_function(fem_results)
            # apply observation function
            o = self._apply_observation_function(fem_results)
            done = self._is_done(fem_results[0]) or self._fem_wrapper.is_terminal_state(simulation_id)

            if self._state_log is not None:
                # create log entry
                raw_data = np.append(self._get_state_vector(fem_results), r)
                # append log
                with open(self._state_log, 'a') as log_handle:
                    np.savetxt(log_handle, raw_data, newline=",")
                    log_handle.write(f',{simulation_id}\n')

        if self.time_step == 0:
            self._root_simulation_id = simulation_id
        self._base_simulation_id = simulation_id

        if done:
            self.info['sim_id'] = simulation_id
            episode_string = f'{self.episode}: Reward {r}, Trajectory {simulation_id}'
            # print(colorize(episode_string, 'green', bold=True))
            logging.info(episode_string)
        self.info['episode'] = self.episode
        self._fem_wrapper.release_lock(simulation_id)

        self.time_step += 1
        return o, r, done, self.info

    def _calc_sim_error_reward(self):
        return self._SIM_ERROR_REWARD

    def _apply_reward_function(self, fem_results):
        """ to be implemented by special FEMEnv instance (use random numbers seeded in seed())
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (object): reward for given simulation results
        """
        raise NotImplementedError

    def _apply_observation_function(self, fem_results):
        """ to be implemented by special FEMEnv instance (use random numbers seeded in seed())
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (object): observation vector for given simulation results
        """
        raise NotImplementedError

    def _sample_process_conditions(self):
        """ to be implemented by special FEMEnv instance (uses random numbers seeded in seed())
        Returns:
            process-conditions (dict):
                dictionary with process-conditions (Keys used have to be identical with abaq-template keys)
        """
        raise NotImplementedError

    def _is_done(self, state=None):
        """ to be implemented by special FEMEnv instance, returns True if the current State is a terminal-state
        Returns:
            done (bool):
                dictionary with process-conditions (Keys used have to be identical with abaq-template keys)
        """
        raise NotImplementedError

    def _get_state_vector(self, fem_results):
        """to be implemented by special FEMEnv instance, returns state values in vector form for fem_results tuple
        Args:
            fem_results: usually an (element-data, node-data) tuple
        Returns:
            state_vec (np.array):
                vector representation of specific fem_results

        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self.simulation_parameters = {}
        self.time_step = 0
        self.info = {}
        self._base_simulation_id = None
        self._root_simulation_id = None
        self.episode += 1

    def render(self, mode='human'):
        """Renders the environment.

        - human: render to the current display or terminal seed and
          return nothing. Usually for human consumption.

        Args:
            mode (str): the mode to render with
        """
        if self.visualize:
            if self.img_ts != self.time_step:
                self.curr_img_path = self._render_state()
                self.img_ts = self.time_step

            if self.curr_img_path is None:
                img = self._standard_img
            else:
                img = imageio.imread(self.curr_img_path, pilmode='RGB')

            if mode == 'rgb_array':
                if img.shape != self._standard_img_size:
                    logging.debug(f'wrong img-shape {img.shape}, expected {self._standard_img_size}')
                    img_ = np.zeros(self._standard_img_size, dtype=np.uint8)
                    img_[:img.shape[0], :img.shape[1], :] = img
                    img = img_
                return img
            if mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1000)
                self.viewer.imshow(img)
                return self.viewer.isopen

    def _render_state(self):
        try:
            state_img_path = self._fem_wrapper.get_state_visualization(time_step=self.time_step)
        except Exception as e:
            warnings.warn(f'problem during visualization {str(e)}')
            return None
        return state_img_path

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:Solver
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self._np_random_state, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class FEMCSVLogger(Wrapper):
    """
    Specific csv-file logger for fem-environments. Complements the default gym monitor / stats_recorder.
    """

    def __init__(self, env, outdir, logmode='per_episode'):
        """
        Args:
            env: fem-environment
            outdir: dir. to log to
            logmode: in {per_episode, per_step} determines the scope of a single log entry.
        """
        assert (type(env) in inspect.getmro(type(env))), \
            f"FEMLogger Wrapper is defined for Environments of type FEMEnv, " \
            f"given: {inspect.getmro(type(env))}"
        assert (logmode in ['per_episode', 'per_step']), 'logmode string has to be one of {per_episode, per_step}'
        self._iteration_start = time.time()
        self._accumulated_reward = 0
        self._logmode = logmode

        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        log_file = outdir.joinpath('env_log.csv')

        if log_file.exists():
            logging.warning(f'{log_file} already existent!')
        self._logger = CSVLogger(log_file)

        # special episode log-file, used when logmode == per_step
        self._episode_logger = None
        if logmode == 'per_step':
            episode_log_file = outdir.joinpath('episode_env_log.csv')
            self._episode_logger = CSVLogger(episode_log_file)
        super().__init__(env)

    def step(self, action):
        o, r, done, info = super().step(action)
        self._accumulated_reward += r

        if self._logmode == 'per_episode':
            try:
                for i, a in enumerate(action):
                    self._logger.set_value(f'action{i}_{self.time_step}', a)
            except TypeError:
                self._logger.set_value(f'action{self.time_step}', action)
            self._logger.set_value(f'reward{self.time_step}', r)
            self._logger.set_values(info)
            if done:
                self._set_episode_log_vals()
                self._logger.write_log()
        else:
            # write step log
            self._logger.set_value('time-step', self.time_step)
            self._logger.set_value(f'reward', r)
            try:
                for i, a in enumerate(action):
                    self._logger.set_value(f'action{i}', a)
            except TypeError:
                self._logger.set_value(f'action{self.time_step}', action)
            self._logger.set_value('done', done)
            self._logger.set_values(info)
            self._logger.write_log()

            # write episode log
            if done:
                self._set_episode_log_vals()
                self._episode_logger.set_value('total_steps', self.time_step)
                if 'episode_goal' in info.keys():
                    self._episode_logger.set_value('goal', info['episode_goal'])
                for k in info.keys():
                    if k.startswith('episode_'):
                        self._episode_logger.set_value(k, info[k])
                self._episode_logger.write_log()

        return o, r, done, info

    def reset(self, **kwargs):
        if self.episode > 0:
            self._accumulated_reward = 0
            self._iteration_start = time.time()
        return self.env.reset(**kwargs)

    def close(self):
        self._logger.write_log()
        return super().close()

    def _set_episode_log_vals(self):
        runtime = time.time() - self._iteration_start
        if self._logmode == 'per_episode':
            self._logger.set_value('iteration', int(self.episode))
            self._logger.set_value('runtime', runtime)
            self._logger.set_value('reward', self._accumulated_reward)
        elif self._logmode == 'per_step':
            self._episode_logger.set_value('episode', int(self.episode))
            self._episode_logger.set_value('runtime', runtime)
            self._episode_logger.set_value('reward', self._accumulated_reward)
