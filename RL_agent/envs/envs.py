import numpy as np
import copy
import warnings
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple, Any, Union

from envs.cell_model import Cell_Population, CellConfig


@dataclass
class EnvConfig:
    num_cells_init: int = 1_000
    k_n0_init: float = 1.0
    b_init: float = 0.0
    n_steps: int = 3_000
    delta_t: float = 0.2 # 0.15 # hours
    threshold: int = 50
    warm_up: int = None
    max_pop: int = int(1e11)
    # max_time: float = 40 # hours

    # observation parameters
    delay_embed_len: int = 20
    k_n0_observation: bool = True
    b_observation: bool = True
    omega: float = 0.02

    num_actions: int = 2
    b_actions: list[int] = field(default_factory=list)

    # constant nutrient environmental parameters
    k_n0_constant: Optional[Union[float, None]] = None
    
    # variable nutrient environmental parameters
    T_k_n0: Optional[Union[float, None]] = None # 6
    k_n0_mean: Optional[Union[float, None]] = None # 2.55
    sigma_kn0: Optional[Union[float, None]] = None # 0.1

    # fluctraing variable nutrient environmental parameters
    T_k_n0_range: list[float] = field(default_factory=list)

    # control of nutrient environmental parameters
    k_n0_actions: Optional[list[float]] = field(default_factory=list)


class BaseEnv(object):
    def __init__(self, env_config, cell_config = CellConfig()):    
        self.env_config = env_config
        self.num_cells_init = env_config.num_cells_init
        self.k_n0_init = env_config.k_n0_init
        self.b_init = env_config.b_init
        self.n_steps = env_config.n_steps
        self.delta_t = env_config.delta_t
        self.threshold = env_config.threshold
        self.warm_up = env_config.warm_up
        self.max_pop = env_config.max_pop
        # self.max_time = env_config.max_time
        self.delay_embed_len = env_config.delay_embed_len
        self.k_n0_observation = env_config.k_n0_observation
        self.b_observation = env_config.b_observation
        self.omega = env_config.omega

        self.num_actions = env_config.num_actions
        self.b_actions = env_config.b_actions

        self.iterations = int(env_config.delta_t * env_config.n_steps)
        if self.warm_up is None:
            warnings.warn("warm_up is not specified, setting warm_up to delay_embed_len.", category=UserWarning)
            self.warm_up = self.delay_embed_len
        elif self.warm_up < self.delay_embed_len:
            warnings.warn("warm_up is less than delay_embed_len, setting warm_up to delay_embed_len.", category=UserWarning)
            self.warm_up = self.delay_embed_len
        self.sim_cells = Cell_Population(cell_config)
        
    def reset(self):
        raise NotImplementedError
    
    def _reset(self):
        self.sim_cells.initialize(self.num_cells_init, self.k_n0_init, self.b_init)
        self.num_cells_history = [0] * self.delay_embed_len
        self.k_n0_history = [0] * self.delay_embed_len
        self.b_history = [0] * self.delay_embed_len

    def step(self, action):
        raise NotImplementedError
    
    def _step(self, k_n0: Union[list, float], b: float):
        if isinstance(k_n0, float) or len(k_n0) == 1:
            k_n0_list = np.ones(self.iterations) * k_n0
        else:
            k_n0_list = k_n0
        
        _, (num_cells_prev, num_cells) = self.sim_cells.simulate_population(k_n0_list, b, self.delta_t, self.n_steps, self.threshold)
        return (self.observation(num_cells_prev, num_cells, k_n0_list[-1], b),
                self.reward(num_cells_prev, num_cells, b),
                self.terminated,
                self.truncated,
                self.info)
    
    def observation(self, num_cells_prev, num_cells, k_n0, b):
        num_cells = 1e-5 if num_cells == 0 else num_cells
        growth_rate = (np.log(num_cells) - np.log(num_cells_prev)) / self.delta_t
        self.num_cells_history.pop(0)
        self.num_cells_history.append(growth_rate)
        self.k_n0_history.pop(0)
        self.k_n0_history.append(k_n0)
        self.b_history.pop(0)
        self.b_history.append(b)
        obs = self.num_cells_history + self.k_n0_history * self.k_n0_observation + self.b_history * self.b_observation
        return copy.deepcopy(obs)
    
    def reward(self, num_cells_prev, num_cells, b):
        num_cells = 1e-5 if num_cells == 0 else num_cells
        growth_rate = (np.log(num_cells) - np.log(num_cells_prev)) / self.delta_t
        cost = growth_rate # + self.omega*b**2 # adding nonlinear penalty for drug application
        return cost

    @property
    def terminated(self):
        return self.sim_cells.true_num_cells == 0
    
    @property
    def truncated(self):
        return self.sim_cells.true_num_cells >= self.max_pop # or self.sim_cells.time_point >= self.max_time
    
    @property
    def info(self):
        return {
            "log": self.sim_cells.logger,
            "delta_t": self.delta_t,
            "warm_up": self.warm_up,
            "delay_embed_len": self.delay_embed_len
        }


class ConstantNutrientEnv(BaseEnv):
    def __init__(self, env_config, cell_config = CellConfig()):
        super().__init__(env_config, cell_config)
        assert len(self.b_actions) == self.num_actions, "Number of actions must match number of antibiotic values"
        assert env_config.k_n0_constant is not None, "k_n0_constant must be specified for constant nutrient environment"
        self.k_n0_constant = env_config.k_n0_constant

        if self.k_n0_init != self.k_n0_constant:
            warnings.warn("k_n0_init does not match k_n0_constant, changing k_n0_init to k_n0_constant.", category=UserWarning)
            self.k_n0_init = self.k_n0_constant
    
    def reset(self) -> tuple:
        self._reset()
        for _ in range(self.warm_up):
            obs, _, _, _, info = self._step(self.k_n0_constant, self.b_init)
        return obs, info
    
    def step(self, action) -> tuple:
        b = self.b_actions[action]
        return self._step(self.k_n0_constant, b)


class VariableNutrientEnv(BaseEnv):
    def __init__(self, env_config, cell_config = CellConfig()):
        super().__init__(env_config, cell_config)
        assert len(self.b_actions) == self.num_actions, "Number of actions must match number of antibiotic values"
        assert env_config.T_k_n0 is not None, "T_k_n0 must be specified for variable nutrient environment"
        assert env_config.k_n0_mean is not None, "k_n0_mean must be specified for variable nutrient environment"
        assert env_config.sigma_kn0 is not None, "sigma_kn0 must be specified for variable nutrient environment"

        self.T_k_n0 = env_config.T_k_n0
        self.k_n0_mean = env_config.k_n0_mean
        self.sigma_kn0 = env_config.sigma_kn0

        if self.k_n0_init != self.k_n0_mean:
            warnings.warn("k_n0_init does not match k_n0_mean, changing k_n0_init to k_n0_mean.", category=UserWarning)
            self.k_n0_init = self.k_n0_mean

    def dkn0dt(self, t, k_n0):
        tau = 3
        Amp = 2

        drift = Amp*np.sin(t*2*np.pi/self.T_k_n0 + self.phase) + self.k_n0_mean
        dkdt = -(1/tau)*(k_n0 - drift)
        return dkdt

    def sim_k_n0(self):
        dt = 1 / self.n_steps
        t = np.linspace(0, self.delta_t, self.iterations) + self.sim_cells.time_point

        k_n0_list = np.zeros(self.iterations)
        k_n0_list[0] = self.k_n0
        for i in range(1, self.iterations):
            k_n0_list[i] = k_n0_list[i-1] + self.dkn0dt(t[i-1], k_n0_list[i-1])*dt + np.sqrt(2*self.sigma_kn0)*np.sqrt(dt)*np.random.normal()
            k_n0_list[i] = np.clip(k_n0_list[i], 0.1, 5.0)
        # k_n0_list = np.clip(k_n0_list, 0.1, 5.0) # clipping values to keep in physiological range
        self.k_n0 = k_n0_list[-1]

        return k_n0_list

    def reset(self) -> tuple:
        self._reset()
        self.k_n0 = self.k_n0_mean
        self.phase = np.random.uniform(0, self.T_k_n0)
        for _ in range(self.warm_up):
            obs, _, _, _, info = self._step(self.sim_k_n0(), self.b_init)
        return obs, info

    def step(self, action) -> tuple:
        b = self.b_actions[action]
        return self._step(self.sim_k_n0(), b)


class ControlNutrientEnv(BaseEnv):
    def __init__(self, env_config, cell_config = CellConfig()):
        super().__init__(env_config, cell_config)
        self.k_n0_actions = env_config.k_n0_actions
        b_num_actions = len(self.b_actions)
        k_n0_num_actions = len(self.k_n0_actions)
        assert b_num_actions * k_n0_num_actions == self.num_actions, "Number of actions must match number of antibiotic values and nutrient values"
        b_mg, k_n0_mg = np.meshgrid(range(b_num_actions), range(k_n0_num_actions))
        self.b_index = b_mg.flatten()
        self.k_n0_index = k_n0_mg.flatten()
    
    def reset(self) -> tuple:
        self._reset()
        for _ in range(self.warm_up):
            obs, _, _, _, info = self._step(self.k_n0_init, self.b_init)
        return obs, info
    
    def step(self, action) -> tuple:
        k_n0 = self.k_n0_actions[self.k_n0_index[action]]
        b = self.b_actions[self.b_index[action]]
        return self._step(k_n0, b)
