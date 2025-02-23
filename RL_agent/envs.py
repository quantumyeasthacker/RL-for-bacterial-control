import numpy as np
import copy
import warnings
from dataclasses import dataclass, field
from cell_model import Cell_Population, CellConfig


@dataclass
class EnvConfig:
    num_cells_init: int = 1_000
    k_n0_init: float = 1.0
    b_init: float = 0.0
    n_steps: int = 3_000
    delta_t: float = 0.15 # hours
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
    k_n0_constant: float = None
    
    # variable nutrient environmental parameters
    T_k_n0: float = 6
    k_n0_mean: float = 2.55
    sigma_kn0: float = 0

    # control of nutrient environmental parameters
    k_n0_actions: list[float] = field(default_factory=list)


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

        self.sim_cells = Cell_Population(cell_config)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
    
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
    
    def info(self):
        return {"log": self.sim_cells.logger,
                "delta_t": self.delta_t,
                "warm_up": self.warm_up,
                "delay_embed_len": self.delay_embed_len}


class ConstantNutrientEnv(BaseEnv):
    def __init__(self, env_config, cell_config = CellConfig()):
        super().__init__(env_config, cell_config)
        assert len(self.b_actions) == self.num_actions, "Number of actions must match number of antibiotic values"
        if env_config.k_n0_constant is None:
            warnings.warn("k_n0_constant not specified, defaulting to k_n0_init.", category=UserWarning)
            # print("Warning: k_n0_constant not specified, defaulting to k_n0_init")
            self.k_n0_constant = self.k_n0_init
        else:
            self.k_n0_constant = env_config.k_n0_constant
    
    def reset(self) -> tuple:
        self.sim_cells.initialize(self.num_cells_init, self.k_n0_init, self.b_init)

        if self.warm_up is not None:
            for _ in range(self.warm_up):
                _, (_, num_cells) = self.sim_cells.simulate_population(self.k_n0_init, self.b_init, self.delta_t, self.n_steps, self.threshold)
        
        self.num_cells_history = []
        self.k_n0_history = [self.k_n0_init] * self.delay_embed_len
        self.b_history = [self.b_init] * self.delay_embed_len
        for _ in range(self.delay_embed_len):
            _, (_, num_cells) = self.sim_cells.simulate_population(self.k_n0_init, self.b_init, self.delta_t, self.n_steps, self.threshold)
            self.num_cells_history.append(num_cells)
        
        obs = self.num_cells_history + self.k_n0_history * self.k_n0_observation + self.b_history * self.b_observation
        return copy.deepcopy(obs), self.info()
    
    def step(self, action) -> tuple:
        b = self.b_actions[action]
        _, (num_cells_prev, num_cells) = self.sim_cells.simulate_population(self.k_n0_constant, b, self.delta_t, self.n_steps, self.threshold)
        return (self.observation(num_cells_prev, num_cells, self.k_n0_constant, b),
                self.reward(num_cells_prev, num_cells, b),
                self.terminated,
                self.truncated,
                self.info())


class VariableNutrientEnv(BaseEnv):
    def __init__(self, env_config, cell_config = CellConfig()):
        super().__init__(env_config, cell_config)
        assert len(self.b_actions) == self.num_actions, "Number of actions must match number of antibiotic values"
        self.T_k_n0 = env_config.T_k_n0
        self.k_n0_mean = env_config.k_n0_mean
        self.sigma_kn0 = env_config.sigma_kn0

    def dkn0dt(self, t, k_n0):
        tau = 3
        Amp = 2

        drift = Amp*np.sin(t*2*np.pi/self.T_k_n0 + self.phase) + self.k_n0_mean
        dkdt = -(1/tau)*(k_n0 - drift)
        return dkdt

    def sim_k_n0(self):
        iterations = int(self.delta_t * self.n_steps)
        dt = 1 / self.n_steps
        t = np.linspace(0, self.delta_t, iterations) + self.sim_cells.time_point

        k_n0_list = np.zeros(iterations)
        k_n0_list[0] = self.k_n0
        for i in range(1,iterations):
            k_n0_list[i] = k_n0_list[i-1] + self.dkn0dt(t[i-1], k_n0_list[i-1])*dt + np.sqrt(2*self.sigma_kn0)*np.sqrt(dt)*np.random.normal()
        k_n0_list = np.clip(k_n0_list, 0.1, 5.1) # clipping values to keep in physiological range
        self.k_n0 = k_n0_list[-1]

        return k_n0_list

    def reset(self) -> tuple:
        self.k_n0 = self.k_n0_mean
        self.phase = np.random.uniform(0,self.T)

        self.sim_cells.initialize(self.num_cells_init, self.k_n0, self.b_init)

        if self.warm_up is not None:
            for _ in range(self.warm_up):
                _, (_, num_cells) = self.sim_cells.simulate_population(self.sim_k_n0(), self.b_init, self.delta_t, self.n_steps, self.threshold)
        
        self.num_cells_history = []
        self.k_n0_history = []
        self.b_history = [self.b_init] * self.delay_embed_len
        for _ in range(self.delay_embed_len):
            _, (_, num_cells) = self.sim_cells.simulate_population(self.sim_k_n0(), self.b_init, self.delta_t, self.n_steps, self.threshold)
            self.num_cells_history.append(num_cells)
            self.k_n0_history.append(self.k_n0)
        
        obs = self.num_cells_history + self.k_n0_history * self.k_n0_observation + self.b_history * self.b_observation
        return copy.deepcopy(obs), self.info()

    def step(self, action) -> tuple:
        b = self.b_actions[action]
        _, (num_cells_prev, num_cells) = self.sim_cells.simulate_population(self.sim_k_n0(), b, self.delta_t, self.n_steps, self.threshold)
        return (self.observation(num_cells_prev, num_cells, self.k_n0, b),
                self.reward(num_cells_prev, num_cells, b),
                self.terminated,
                self.truncated,
                self.info())


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
        self.sim_cells.initialize(self.num_cells_init, self.k_n0_init, self.b_init)

        if self.warm_up is not None:
            for _ in range(self.warm_up):
                _, (_, num_cells) = self.sim_cells.simulate_population(self.k_n0_init, self.b_init, self.delta_t, self.n_steps, self.threshold)
        
        self.num_cells_history = []
        self.k_n0_history = [self.k_n0_init] * self.delay_embed_len
        self.b_history = [self.b_init] * self.delay_embed_len
        for _ in range(self.delay_embed_len):
            _, (_, num_cells) = self.sim_cells.simulate_population(self.k_n0_init, self.b_init, self.delta_t, self.n_steps, self.threshold)
            self.num_cells_history.append(num_cells)

        
        obs = self.num_cells_history + self.k_n0_history * self.k_n0_observation + self.b_history * self.b_observation
        return copy.deepcopy(obs), self.info()
    
    def step(self, action) -> tuple:
        k_n0 = self.k_n0_actions[self.k_n0_index[action]]
        b = self.b_actions[self.b_index[action]]
        _, (num_cells_prev, num_cells) = self.sim_cells.simulate_population(k_n0, b, self.delta_t, self.n_steps, self.threshold)
        return (self.observation(num_cells_prev, num_cells, k_n0, b),
                self.reward(num_cells_prev, num_cells, b),
                self.terminated,
                self.truncated,
                self.info())