import hashlib
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shutil
import subprocess
import time
import warnings
from abc import ABC
from datetime import datetime
from math import pi
from pathlib import Path
from pymks.bases import GSHBasis
from pyquaternion import Quaternion
from scipy import spatial
from scipy.spatial.transform import Rotation
from subprocess import check_output

from msevolution_env.helpers import CSVLogger


class FEMWrapper(ABC):
    """ superclass for FEM-software specific wrapper-classes
    """

    def __init__(self, simulation_store_path):
        self.simulation_store_path = simulation_store_path

    def run_simulation(self, simulation_id, simulation_parameters, time_step, base_simulation_id=None):
        """ executes the simulation with given parameters and stores results under the given simulation-id
        Args:
            simulation_id (str): identifier of the requested simulation
            simulation_parameters (dict): named parameter-values for the parametric fem-simulation
            base_simulation_id (str): identifier of the basis simulation for the requested simulation
            time_step (int): current time-step
        """
        raise NotImplementedError

    def simulation_results_available(self, simulation_id):
        """ Returns true if given simulation has been calculated and results are available
        Args:
            simulation_id (str): identifier of the requested simulation
        Returns:
            results_available (bool): true if results are available, else false
        """
        raise NotImplementedError

    def read_simulation_results(self, simulation_id, root_simulation_id=None):
        """ Reads out results from the given simulation
        Args:
            simulation_id (str): identifier of the requested simulation
            root_simulation_id (str): identifier of the root simulation (first time-step)
        Returns:
            simulation_results (tuple): Tuple of two Pandas Dictionaries: (element-wise results, node-wise results)
        """
        raise NotImplementedError

    def is_terminal_state(self, simulation_id):
        """ returns True, if no former actions are possible based on the simulation-state
        Args:
            simulation_id (str): identifier of the requested simulation
        Returns:
            is_terminal (bool): True, if no former actions are possible based on the simulation-state
        """
        return False

    def request_lock(self, simulation_id, timeout_seconds=600):
        """ locks the given simulation (to enable parallel environments based on the same simulation-storage)
        Args:
            simulation_id (str): identifier of the simulation
            timeout_seconds (int): seconds from now until lock times out
        Returns:
            locked (bool): True if the lock has been successfully established, false if the lock was already set
        """
        job_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
        job_directory.mkdir(parents=True, exist_ok=True)

        lock = job_directory.joinpath('.fem_wrapper_lock')
        try:
            lock.touch(exist_ok=False)
        except FileExistsError:
            lock_timeout = lock.read_text()
            if lock_timeout == '':
                logging.warning(f'empty lock: {lock}, setting lock-timeout t+600 and waiting')
                timeout = str(datetime.now().timestamp() + timeout_seconds)
                lock.write_text(timeout)
                return False
            elif datetime.now().timestamp() > float(lock_timeout):
                logging.warning(
                    f'overwriting timed out lock: {lock}, timeout: {datetime.fromtimestamp(float(lock_timeout))}')
            else:
                return False

        timeout = str(datetime.now().timestamp() + timeout_seconds)
        lock.write_text(timeout)
        return True

    def release_lock(self, simulation_id):
        """ Releases the simulation-lock
        Args:
            simulation_id (str): identifier of the requested simulation
        """
        job_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
        lock = job_directory.joinpath('.fem_wrapper_lock')
        if lock.exists():
            lock.unlink()


class SimulationError(Exception):
    """ Thrown if simulation is running into numeric issues for the given parameters
    """

    def __init__(self):
        self.message = "Simulation-Model can not be solved for given parameters"


class MStructEvoWrapper(FEMWrapper):
    """
    proprietary wrapper for the microstructure-evolution simulation
    """
    microstructures_path = Path(__file__).parent.joinpath(f'assets/microstructures')

    def __init__(self, simulation_store_path, material='100ud_oris.inp',
                 cpu_kernels=40, terminal_state_plot_folder=None, simulation_timeout=600):
        """
        Args:
            simulation_store_path:
            goal_microstructure_file:
            material:
            cumulative_rotations:
            cpu_kernels:
            terminal_state_plot_folder:
        """

        super().__init__(simulation_store_path)
        self.simulation_timeout = simulation_timeout
        self.sim_path = Path(__file__).parent.joinpath(f'assets/sim/')

        self.sim_result_files = ['INFO_uniax_simulator.dat',
                                 'OUT_dfgrd1.dat',
                                 'OUT_microstructure_ori_list.dat',
                                 'OUT_statev.dat',
                                 'IN_actual_load.inp']
        self.cpu_kernels = cpu_kernels
        # ------------------------- set material --------------------------
        shutil.copy(self.microstructures_path.joinpath(material), self.sim_path.joinpath('mat.inp'))

        # internal variables for persisting final state images
        self.terminal_state_plot_folder = terminal_state_plot_folder
        self._curr_image_path = None
        self._episode = 0

        self._curr_goal = None
        self._curr_eps_eq = 0
        sim_log_file = self.simulation_store_path.joinpath('sim_log.csv')
        self.csv_logger = CSVLogger(sim_log_file)

        self.MAX_EPS_EQ = 0.7

    def run_simulation(self, simulation_id, simulation_parameters, time_step, base_simulation_id=None):
        """ executes the simulation with given parameters and stores results under the given simulation-id
        Args:
            simulation_id (str): identifier of the requested simulation
            simulation_parameters (dict): named parameter-values for the parametric fem-simulation
            base_simulation_id (str): identifier of the basis simulation for the requested simulation
            time_step (int): current time-step
        """
        job_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
        job_directory.mkdir(parents=True, exist_ok=True)

        n_tasks = self.cpu_kernels

        self.csv_logger.set_value('ts', time_step)
        # log: parameters, run-time, exit code

        # write actual_load.inp (direction, load)
        inp_load_file = self.sim_path.joinpath('IN_actual_load.inp')

        rotation_params = [f'{p}_{time_step}' for p in ['Q1', 'Q2', 'Q3', 'Q4']]
        rotation_q = Quaternion([str(simulation_parameters[p]) for p in rotation_params])

        # build and write input_load string
        restart_str = '0.0' if time_step == 0 else '1.0'
        displacement = simulation_parameters[f'd_{time_step}']
        inp_str = f"{restart_str}, 1.0, {displacement}, " + ', '.join([str(qe) for qe in rotation_q.elements])
        inp_load_file.write_text(inp_str)

        log_str = np.array_str(np.append([displacement], rotation_q.elements), precision=5,
                               suppress_small=True).replace('\n', '')
        self.csv_logger.set_value('parameters', log_str)

        # execute simulation (if timestep == 0 init)
        if time_step == 0:
            # clean
            out_files = ['INFO_uniax_simulator.dat', 'OUT_dfgrd1.dat', 'OUT_microstructure_ori_list.dat',
                         'OUT_statev.dat']
            for f in out_files:
                f = self.sim_path.joinpath(f)
                if f.exists():
                    f.unlink()
        t0 = time.time()
        self.csv_logger.set_value('sim_dir', job_directory)

        try:
            check_output([f'./uniax_simulator_for_microstructure_evolution_{n_tasks}tasks'],
                         cwd=str(self.sim_path), timeout=self.simulation_timeout)
        except subprocess.TimeoutExpired:
            self.csv_logger.set_value('sim_time', time.time() - t0)
            self.csv_logger.set_value('exit_code', 99)
            self.csv_logger.write_log()
            warnings.warn('Simulation timeout!')

            shutil.copy(str(self.sim_path.joinpath('INFO_uniax_simulator.dat')),
                        str(job_directory.joinpath('INFO_uniax_simulator.dat')))
            raise SimulationError
        except subprocess.CalledProcessError as e:
            self.csv_logger.set_value('sim_time', time.time() - t0)
            self.csv_logger.set_value('exit_code', e.returncode)
            self.csv_logger.write_log()

            # check for convergence error
            if e.returncode in [51, 52]:
                shutil.copy(str(self.sim_path.joinpath('INFO_uniax_simulator.dat')),
                            str(job_directory.joinpath('INFO_uniax_simulator.dat')))
                raise SimulationError
            elif e.returncode == 127:
                raise EnvironmentError(f'error during microstructure-evolution simulation.\n'
                                       f'Intel Fortran compiler has to be installed and LD_LIBRARY_PATH'
                                       f' has to be set in order to run the simulation {str(e)}')

        self.csv_logger.set_value('sim_time', time.time() - t0)
        print('sim_time', time.time() - t0)
        # copy results (OUT_xxx.dat) to simulation-folder
        job_directory.mkdir(parents=True, exist_ok=True)
        for f in self.sim_result_files:
            shutil.copy(str(self.sim_path.joinpath(f)), str(job_directory.joinpath(f)))
        dfgrd1 = np.loadtxt(job_directory.joinpath('OUT_dfgrd1.dat'))
        self.csv_logger.set_value('dfgrd',
                                  np.array_str(dfgrd1.flatten(), precision=5, suppress_small=True).replace(
                                      '\n', ''))
        # check dfgrd condition
        eps_eq = self.calc_eps_eq(dfgrd1)
        self._curr_eps_eq = eps_eq
        print(f'eps-eq: {eps_eq}')
        self.csv_logger.set_value('eps_eq', eps_eq)
        self.csv_logger.set_value('exit_code', 0)
        self.csv_logger.write_log()

        if eps_eq > self.MAX_EPS_EQ:
            self.csv_logger.set_value('exit_code', 55)
            warnings.warn(f'equivalent strain to high: {eps_eq}')
            raise SimulationError

    def get_eps_eq(self):
        return self._curr_eps_eq

    def simulation_results_available(self, simulation_id):
        """ Returns true if given simulation has been calculated and results are available
        Args:
            simulation_id (str): identifier of the requested simulation
        Returns:
            results_available (bool): true if results are available, else false
        """
        job_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
        return job_directory.joinpath('INFO_uniax_simulator.dat').exists()

    def read_simulation_results(self, simulation_id, root_simulation_id=None):
        """ Reads out results from the given simulation
        Args:
            simulation_id (str): identifier of the requested simulation
            root_simulation_id (str): identifier of the root simulation (first time-step)
        Returns:
            simulation_results (ODF):
        """
        if simulation_id == 'initial':
            # return microstructure for mat.inp
            mat_in_file = self.sim_path.joinpath('mat.inp')
            odf = ODF.from_mat_file(mat_in_file)
        else:
            job_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
            sim_out_file = job_directory.joinpath('OUT_microstructure_ori_list.dat')
            if sim_out_file.exists():
                odf = ODF.from_ori_file(sim_out_file, weights=False)
            else:
                odf = None
        return odf, None

    def is_terminal_state(self, simulation_id):
        job_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
        if job_directory.exists():
            dfgrd1 = np.loadtxt(job_directory.joinpath('OUT_dfgrd1.dat'))
            eps_eq = self.calc_eps_eq(dfgrd1)
            if eps_eq > self.MAX_EPS_EQ:
                warnings.warn(f'equivalent-strain to high: {eps_eq}')
                return True
        return False

    def get_state_visualization(self, time_step, odf=None, goal_odf=None,
                                distance_measure='chi_square'):
        try:
            # todo extract pole-figure code
            from TX import upf
        except ModuleNotFoundError:
            warnings.warn(
                f'texture package is required for polefigure visualization!\nclone and install from:\nhttps://github.com/johannes-dornheim/texture\nhttps://github.com/johannes-dornheim/mpl-lib')
            return None
        matplotlib.use('Agg')
        """
        if simulation_id is not None:
            visualization_directory = self.simulation_store_path.joinpath(f'simulations/{simulation_id}')
            img_path = visualization_directory.joinpath('vis.jpg')
            microstructure_q = self._read_odf_from_simulation_output(
                visualization_directory.joinpath('OUT_microstructure_ori_list.dat'))
            pf_microstructure = self._quats_to_pf_eulers(microstructure_q)
        """
        if odf is not None:
            visualization_directory = self.simulation_store_path.joinpath('state_visualizations')
            visualization_directory.mkdir(parents=True, exist_ok=True)
            img_path = visualization_directory.joinpath(f'vis_{time_step}.jpg')
        else:
            warnings.warn('Either state ODF or simulation-id needed for state-visualization')
            return None

        # plot goalfig
        goalfig_path = self.sim_path.joinpath('goal_polefigs.png')  # todo regard multi-goal setting
        if self._curr_goal != goal_odf:
            gf = goal_odf.create_pole_plots()
            gf.savefig(goalfig_path, bbox_inches='tight')
            self._curr_goal = goal_odf
            plt.close(gf)
        img = plt.imread(str(goalfig_path))

        # plot polefigure

        pf = odf.create_pole_plots()
        # append figure by histogram plot (kind of abuse of gridspec, but I dont want to alter pf_new code)
        pf.set_size_inches(14, 32)
        gs = pf.add_gridspec(nrows=100, ncols=1)

        # show goal-polefig
        gp = pf.add_subplot(gs[50:75, 0])
        gp.axis('off')
        plt.imshow(img)

        pf.add_subplot(gs[70:101, 0])

        hist = odf.get_whd()
        goal_hist = goal_odf.get_whd()
        width = 0.35
        hist_size = len(goal_hist)
        plt.bar(np.arange(hist_size) + width, goal_hist, width, label='goal microstructure')
        plt.bar(np.arange(hist_size), hist, width, label='microstructure')
        plt.ylim(top=max(max(goal_hist) * 1.4, max(hist) * 1.05))
        plt.legend()
        d = odf.get_distance(goal_odf, distance_measure) * 100
        plt.title(f"t.: {time_step}, d.: {round(d, 5)}, eps-eq.: {round(self._curr_eps_eq, 5)}")

        pf.savefig(img_path, bbox_inches='tight')
        plt.clf()
        plt.close(pf)

        if self.terminal_state_plot_folder is not None and \
                time_step == 0 and \
                self._curr_image_path is not None:
            if not Path(self.terminal_state_plot_folder).exists():
                Path(self.terminal_state_plot_folder).mkdir(parents=True, exist_ok=True)
            goal_path = Path(self.terminal_state_plot_folder).joinpath(f'final_state_eps{self._episode}.jpg')
            shutil.copy(self._curr_image_path, goal_path)
            self._episode += 1
        self._curr_image_path = img_path

        return img_path

    @classmethod
    def calc_eps_eq(cls, dfgrd1):
        # Calculate Green strain tensor
        E = cls._get_Green_strain_tensor(dfgrd1)

        # Calculate principle strains
        E_PC = cls._get_principle_components(E)

        # Apply von Mises equivalent strain
        return cls._get_vonMises_equivalent_strain(E_PC)

    @staticmethod
    def _get_Green_strain_tensor(F):
        return 0.5 * (np.dot(np.transpose(F), F) - np.eye(3))

    @staticmethod
    def _get_principle_components(A):
        PC, _ = np.linalg.eig(A)
        return PC

    @staticmethod
    def _get_vonMises_equivalent_strain(E_PC):
        """ van Mises equivalent strain calculated based on principle strains"""
        return np.sqrt(2.0 / 3.0 * (np.square(E_PC[0]) + np.square(E_PC[1]) + np.square(E_PC[2])))


_UD_ORIS_KD_TREE = {}
_ASSETS = Path(__file__).parent.joinpath('assets')


class ODF:
    ORIS = None
    WEIGHTS = None
    whd_weights = None
    gsh_coeffs = None
    _polefig = None
    _hash = None

    def __init__(self, orientations, weights=None, whd_soft_assignment=3, symmetry='cubic', sample_symmetry='triclinic',
                 whd_bins=512, gsh_degree=8):
        """
        Args:
            orientations (List[Quaternion]): list of pyquaternion Quaternion objects
            weights (np.Array[float]): list  of weights (float). has to be the same length as the orientations list
            whd_soft_assignment (int): count of bins used to represent a single orientation
            symmetry (str): crystal-symmetry
            whd_bins (int): count of bins (spatial discretization)
        """

        self.ORIS = orientations
        if weights is None:
            self.WEIGHTS = np.ones(len(orientations))
        else:
            assert len(weights) == len(orientations)
            self.WEIGHTS = weights
        h = hashlib.sha1(str.encode(f'{str(orientations)} {str(weights)}')).hexdigest()
        self._hash = int(h, 16)

        self.whd_weights = None
        self.whd_bin_oris = None
        self.gsh_coeffs = None

        self._whd_soft_assignment = whd_soft_assignment
        self._n_whd_bins = whd_bins
        self._SYMMETRY = symmetry
        self._SAMPLE_SYMMETRY = sample_symmetry
        self._gsh_degree = gsh_degree

        # create KDTree if not already done
        bin_ori_file_by_bin_count = {n: f'ud_oris/n{n}-id1_quaternion.ori' for n in
                                     [100, 128, 256, 512, 1024, 2048, 4096, 8192]}
        if whd_bins in bin_ori_file_by_bin_count.keys() and symmetry in ['triclinic', 'cubic']:
            global _ASSETS
            global _UD_ORIS_KD_TREE
            # declare tree only once
            if whd_bins not in _UD_ORIS_KD_TREE.keys():
                # read in uniform_orientations (uo)
                ud_ori_file = _ASSETS.joinpath(bin_ori_file_by_bin_count[whd_bins])
                self.whd_bin_oris = [Quaternion(o) for o in np.loadtxt(ud_ori_file)]
                ud_orientations = self._get_equivalent_quaternions(self.whd_bin_oris, symmetry=symmetry)

                # build kd-tree
                candidate_elems = [x.elements for x in ud_orientations]
                _UD_ORIS_KD_TREE[whd_bins] = spatial.cKDTree(candidate_elems)
            else:
                ud_ori_file = _ASSETS.joinpath(bin_ori_file_by_bin_count[whd_bins])
                self.whd_bin_oris = [Quaternion(o) for o in np.loadtxt(ud_ori_file)]
        else:
            raise NotImplementedError

    # ==================================== object creation ====================================
    @classmethod
    def from_quaternions(cls, quats, **args):
        """
        Args:
            quats (List[Quaternion]):
            **args:
        Returns:
        """
        if type(quats[0]) in [float, np.float64]:
            quats = [Quaternion(q) for q in quats.reshape(-1, 4)]
        return cls(quats, **args)

    @classmethod
    def from_ori_file(cls, ori_file, weights=False, euler_degrees=True, delimiter=None, **args):
        """ generate Microstructure object from files in the simulation-output format (one line per ori, tab-separated)
        Args:
            ori_file (Path): Path to simulation output file
            weights: ODF weights
            euler_degrees: orientation-encoding flag
            delimiter: alternative delimiter
        Returns:
            odf (Microstructure): List of orientations as Quaternion Objects
        """
        if type(ori_file) == str:
            ori_file = Path(ori_file)
        assert ori_file.exists(), f'{ori_file} not existant!'

        orientations = np.loadtxt(str(ori_file), ndmin=2, delimiter=delimiter)
        if not weights:
            if orientations.shape[1] == 3:
                orientations_q = cls._bunge_eulers_to_quats(orientations, degrees=euler_degrees)
            elif orientations.shape[1] == 4:
                # quat
                orientations_q = [Quaternion(real=q[0], imaginary=q[1:4]) if not np.array_equal(q, np.zeros(4))
                                  else Quaternion(real=1, imaginary=[0., 0., 0.]) for q in orientations]
            else:
                raise Exception(f'output file {ori_file} is malformed!')
            return cls.from_quaternions(orientations_q, **args)

        else:
            if orientations.shape[1] == 4:
                orientations_q = cls._bunge_eulers_to_quats(orientations[:, :-1], degrees=euler_degrees)
                weights = orientations[:, -1]

            elif orientations.shape[1] == 5:
                # quat
                orientations_q = [Quaternion(real=q[0], imaginary=q[1:4]) if not np.array_equal(q, np.zeros(4))
                                  else Quaternion(real=1, imaginary=[0., 0., 0.]) for q in orientations[:, :-1]]
                weights = orientations[:, -1]
            else:
                raise Exception(f'output file {ori_file} is malformed!')

            return cls.from_quaternions(orientations_q, weights=weights, **args)

    @classmethod
    def from_mat_file(cls, mat_inp_file, **args):
        """ import odf from mat.inp file
        Args:
            mat_inp_file (Path): Path to mat.in file
        Returns:
            odf (list(Quaternion)): List of orientations as Quaternion Objects
        """
        if type(mat_inp_file) == str:
            mat_inp_file = Path(mat_inp_file)

        mat_in = mat_inp_file.read_text().split('\n')[1:]

        orientations = [np.float_(s.split(',')[:3]) for i, s in enumerate(mat_in)]
        orientations = np.array(orientations)
        orientations_q = cls._bunge_eulers_to_quats(orientations)

        return cls.from_quaternions(orientations_q, **args)

    def get_oris(self):
        return self.ORIS

    def get_whd(self):
        if self.whd_weights is None:
            self.whd_weights = self._build_microstructure_hist()
        return self.whd_weights

    def get_gsh(self, as_float=True):
        if self.gsh_coeffs is None:
            odf_bunge = self._quats_to_bunge_eulers(self.ORIS, degrees=False)
            odf_bunge_radian_w_weights = np.hstack([odf_bunge, self.WEIGHTS.reshape(-1, 1)])
            self.gsh_coeffs = self._descriptor_gsh(odf_bunge_radian_w_weights,
                                                   self._SYMMETRY, self._SAMPLE_SYMMETRY, self._gsh_degree)
        if as_float:
            return np.array([[c.real, c.imag] for c in self.gsh_coeffs]).flatten()
        else:
            return self.gsh_coeffs

    def get_distance(self, other, dst='whd_chi_square'):
        hist1 = self.get_whd()
        hist2 = other.get_whd()
        assert len(hist1) == len(hist2), "WHD Distance can only be measured for ODFs with Same whd-bin-count"
        if dst == 'whd_chi_square':
            return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 0.0000000001))
        else:
            raise NotImplementedError

    def _get_equivalent_quaternions(self, quaternions, symmetry='cubic'):
        """
        Args:
            quaternions (List[Quaternion]): list of quaternions
            symmetry (Str): symmstry type (currently suppoerted: 'cubic'
        Returns:
            list of equivalent quaternions for the given input quaternions
        """
        if symmetry == 'cubic':
            # equivalence operations for cubic symmetry
            # identity
            equivs = [Quaternion(1, 0, 0, 0)]
            # <1,0,0> eqivs (9)
            equivs += [Quaternion(axis=[1, 0, 0], angle=i * pi) for i in [0.5, 1, 1.5]] + \
                      [Quaternion(axis=[0, 1, 0], angle=i * pi) for i in [0.5, 1, 1.5]] + \
                      [Quaternion(axis=[0, 0, 1], angle=i * pi) for i in [0.5, 1, 1.5]]
            # <1,1,0> equivs (6)
            equivs += [Quaternion(axis=a, angle=pi) for a in [[1, 1, 0], [1, 0, 1], [0, 1, 1],
                                                              [-1, 1, 0], [-1, 0, 1], [0, -1, 1]]]
            # <1,1,1> eqivs (8)
            equivs += [Quaternion(axis=[1, 1, 1], angle=i * pi) for i in [2 / 3, -2 / 3]] + \
                      [Quaternion(axis=[1, 1, -1], angle=i * pi) for i in [2 / 3, -2 / 3]] + \
                      [Quaternion(axis=[1, -1, 1], angle=i * pi) for i in [2 / 3, -2 / 3]] + \
                      [Quaternion(axis=[-1, 1, 1], angle=i * pi) for i in [2 / 3, -2 / 3]]
        elif symmetry == 'triclinic':
            equivs = [Quaternion(1, 0, 0, 0)]
        elif symmetry == 'orthorhombic':
            equivs = [Quaternion(1, 0, 0, 0)]
            equivs += [Quaternion(axis=[0, 1, 0], angle=i * pi) for i in [0.5, 1., 1.5]]
        else:
            raise NotImplementedError
        q_equivs = []
        for o in quaternions:
            q_equivs += [o * q for q in equivs]
        q_equivs = np.append(np.array(q_equivs), -np.array(q_equivs))
        return q_equivs

    @staticmethod
    def _bunge_eulers_to_quats(bunge_eulers, degrees=True):
        """
        Args:
            bunge_eulers: list of Bunge euler angles (ZXZ)
        Returns:
            quats (list(Quaternion)): converted orientations as Quaternion object list
        """
        orientations_q = Rotation.from_euler('ZXZ', bunge_eulers, degrees=degrees).as_quat()
        return [Quaternion(real=q[3], imaginary=q[0:3]) for q in orientations_q]

    @staticmethod
    def _soft_assign(neighbor_idxs, distances, bin_count, prior_weights):
        """
        Args:
            neighbor_idxs (np.Array): list of neighbor indices (as created by scipy kdTree)
            distances (np.Array): list of distances to indices (as created by scipy kdTree)
            bin_count (int): count of bins of the resulting hist
        Returns:
            soft assigned histogram with 'bin_count' bins
        """
        if len(neighbor_idxs.shape) == 1:
            neighbor_idxs = neighbor_idxs.reshape(-1, 1)
            distances = distances.reshape(-1, 1)

        # reciprocals, summing to 1
        reciprocals = 1 / (distances + 1e-10)

        weights = reciprocals / reciprocals.sum(axis=1, keepdims=1)
        weights = weights * prior_weights[:, np.newaxis]
        hist = np.array([weights[neighbor_idxs == i].sum() for i in np.arange(bin_count)])
        return hist

    @staticmethod
    def _quats_to_bunge_eulers(quats, degrees=True):
        return np.array([Rotation.from_quat(q.elements[[1, 2, 3, 0]]).as_euler('ZXZ', degrees=degrees) for q in quats])

    def _build_microstructure_hist(self):
        global _UD_ORIS_KD_TREE
        odf_elems = [x.elements for x in self.ORIS]
        dists, nns = _UD_ORIS_KD_TREE[self._n_whd_bins].query(odf_elems, self._whd_soft_assignment, n_jobs=-1)

        # invert symmetries
        nns = (nns // 24) % self._n_whd_bins

        soft_hist = self._soft_assign(nns, dists, self._n_whd_bins, self.WEIGHTS)
        # return normalized hist
        return soft_hist / sum(soft_hist)

    @classmethod
    def _descriptor_gsh(cls, microstructure, cry_sym, sample_symmetry, L):
        """ Generalized spherical harmonics descriptor based on the implementation in pymks [1]
        ------------------------------------------------------------------------------------------------------------------------
        copied from descriptor_gsh.py

        Generalized spherical harmonics descriptor using pymks framework [1] (http://pymks.org/en/latest/rst/README.html)

        Author:mor
        Version 2019-07-22
        ------------------------------------------------------------------------------------------------------------------------

            INPUT:  microstructure  ... microstructure (np.array) as odfs in Bunge euler angle format [radians]
                    cry_sym         ... crystal symmetry
                    samp_sym        ... sample symmetry
                    L               ... degree, where to cut gsh expansion
            OUTPUT: gsh_coefs       ... generalized spherical harmonics coefficients
        """

        if sample_symmetry == 'triclinic':
            # Constant parameters
            n_states = cls._n_coefs_for_L(cry_sym, sample_symmetry, L)

            # Init gsh coefficients array
            gsh_coefs = np.zeros(n_states, dtype=np.complex64)

            # Load HSG basis
            gsh_basis = GSHBasis(n_states, domain=cry_sym)

            # Getting GSH coefficients as sum over orientations and their weights as described in [2] eq. (4.20)
            weights = microstructure[:, 3]
            microstructure_disc = gsh_basis.discretize(microstructure[:, :3])
            gsh_coefs[:] = np.sum((microstructure_disc.T * weights).T, axis=0) / np.sum(weights)

            # For machine learning mostly half of the gsh coefficients can be exluded [3] p.2650
            gsh = cls._reduce_gsh_coefs(gsh_coefs, n_states)

        return gsh

    @classmethod
    def _n_coefs_for_L(cls, cry_sym, samp_sym, L, step=1):
        """ Gives back the number of states for pymks's gsh series expansion implementation to fully describe up to a certain degree L.
            INPUT:  cry_sym   ... crystal symmetry
                    samp_sym  ... sample symmetry
                    L         ... Degree to cut the gsh series expansion
            OUTPUT: n_states  ... Number of coefficients of pymks to fully describe the given degree
        """

        # Sum over linearly independent harmonics, see [2] Fig. 4.4 and 14.1
        n_states = 0
        for l in range(0, L + 1, step):
            n_states += cls._number_lin_indepent_harm(cry_sym, l) * cls._number_lin_indepent_harm(samp_sym, l)

        return int(n_states)

    @classmethod
    def _number_lin_indepent_harm(cls, symmetry, L):
        """ Returns number of linear independet harmonics w.r.t. symmetry and degree of harmonics L. See [2] Fig. 4.4 and 14.1
            INPUT:  symmetry ... crystal or sample symmetry
                    L        ... degree of harmonics
            OUTPUT: M        ... Number of linearly independent harmonics at degree L and for a given symmetry
        """

        if symmetry == 'triclinic':  # triclinic point group S_2
            M = 2 * L + 1
        elif symmetry == 'cubic':  # cubic point group O_h
            if L in [1, 2, 3, 5, 7, 11]:
                M = 0
            elif L in [0, 4, 6, 8, 9, 10, 13, 14, 15, 17, 19, 23]:
                M = 1
            elif L in [12, 14, 16, 18, 20, 21, 22, 25, 26, 27, 29, 31, 35]:
                M = 2
            elif L in [24, 28, 30, 32, 33, 34, 37, 38, 39, 41, 43, 47]:
                M = 3
            elif L in [36, 40, 42, 44, 45, 46, 49, 50]:
                M = 4
            elif L in [48]:
                M = 5
            else:
                raise RuntimeError(
                    'Number of linearly independent harmoncis cannot be given as the specified degree is above 50, which is not implemented.')
        else:
            raise RuntimeError(
                'No number of linearly independent harmonics can be given for the specified symmetry, as it is not implemented.')

        return int(M)

    @classmethod
    def _reduce_gsh_coefs(cls, gsh_coefs, n_states):
        """ Reduces redundancies in GSH-coefficients that do not need to be considered for ML, see [3] p.2650.
            INPUT:  gsh_coefs ... GSH coefficients from function <descriptor_gsh>
                    cry_sym   ... crystal symmetry
                    samp_sym  ... sample symmetry
                    n_states  ... Number of coefficients of pymks to fully describe the given degree
            OUTPUT: gsh_red   ... reduced GSH coefficients
        """

        # Get indices of significant coefficents
        indices = np.array(
            [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46,
             47,
             48, 49, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
             94,
             95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
             142,
             143, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 186, 187, 188, 189, 190,
             191,
             192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
             228,
             229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249])
        indices = indices[np.where(indices < n_states)]

        return gsh_coefs[indices]
