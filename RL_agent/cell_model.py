import numpy as np
from scipy import optimize
from dataclasses import dataclass


@dataclass
class CellConfig:
    phiR_min = 0
    phiR_max: float = 0.55 # Scott et al. 2010
    a_n: float = 1e-3 # level to trigger negative feeback inhibition of amino acid supply, Scott et al. 2014
    a_t: float = 1e-4 # amino acid level for efficient peptide elongation, Scott et al. 2014
    n_f: int = 2 # cooperativity in feedback
    n_g: int = 2

    # Kratz and Banerjee 2023
    sigma: float = 0.015 # noise strength
    alphaX: float = 4.5
    betaX: float = 1.1
    mu: float = 0.6
    k_t0: float = 2.7 # translational efficiency
    q: int = 2
    phiS_max: float = 0.33
    alpha: float = 1.54
    beta: float = 10.5
    K_u: float = 0.076


class Cell_Population(object):
    def __init__(self, cell_config):        
        self.true_num_cells = None
        self.populations = None
        self.guess_value_list = None
        self._log = None

        # cell parameters
        self.cell_config = cell_config
        self.phiR_min = cell_config.phiR_min
        self.phiR_max = cell_config.phiR_max
        self.a_n = cell_config.a_n
        self.a_t = cell_config.a_t
        self.n_f = cell_config.n_f
        self.n_g = cell_config.n_g

        # Kratz and Banerjee 2023
        self.sigma = cell_config.sigma
        self.alphaX = cell_config.alphaX
        self.betaX = cell_config.betaX
        self.mu = cell_config.mu
        self.k_t0 = cell_config.k_t0
        self.q = cell_config.q
        self.phiS_max = cell_config.phiS_max
        self.alpha = cell_config.alpha
        self.beta = cell_config.beta
        self.K_u = cell_config.K_u

        # defining regulatory functions and their derivatives
    def f(self, a):
        return 1 / (1 + (a/self.a_n)**self.n_f) # regulatory function for k_n
    def f_prime(self, a):
        return -(self.n_f/self.a_n)*(a/self.a_n)**(self.n_f-1) / (1 + (a/self.a_n)**self.n_f)**2 # derivative of f w.r.t. a
    def g(self, a):
        return (a/self.a_t)**self.n_g / (1 + (a/self.a_t)**self.n_g) # regulatory function for k_t
    def g_prime(self, a):
        return (self.n_g/self.a_t)*(a/self.a_t)**(self.n_g-1) / (1 + (a/self.a_t)**self.n_g)**2 # derivative of g w.r.t. a

    # f_S, fraction of total cell synthesis capcity devoted to stress protein production
    def f_S(self, U):
        return self.phiS_max * (U**self.q / (self.K_u**self.q + U**self.q))
    # f_R, fraction of total cell synthesis capacity devoted to ribosome production
    def f_R(self, a, U):
        return (-self.f_prime(a)*self.g(a)*(self.phiR_max-self.f_S(U)) + self.f(a)*self.g_prime(a)*self.phiR_min) / (-self.f_prime(a)*self.g(a) + self.f(a)*self.g_prime(a))
    # f_X, fraction of cell synthesis capacity devoted to division protein production
    def f_X(self, a, U):
        return self.alphaX*(self.phiR_max - self.f_R(a,U)) + self.betaX

    def U_t(self, U):
        ut = 1 - U
        # return np.maximum(ut, 0)
        return ut.clip(0, None)

    def GrowthRate(self, a, phi_R, U):
        # growth rate function
        k_t = self.k_t0 * self.g(a) * self.U_t(U)
        k = k_t * (phi_R - self.phiR_min)
        return k

    def dphiR_dt(self, phi_R, a, U):
        # ribosome mass fraction ODE
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency
        dpdt = k_t * (phi_R - self.phiR_min) * (self.f_R(a,U) - phi_R)
        return dpdt

    def dphiS_dt(self, phi_S, a, phi_R, U):
        # stress sector mass fraction ODE
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency
        dpdt = k_t * (phi_R - self.phiR_min) * (self.f_S(U) - phi_S)
        return dpdt

    def dAAdt(self, a, phi_R, phi_S, U, k_n0):
        # amino acid concentration ODE (variable nutrient conc.(c))
        k_n = k_n0 * self.f(a) # nutritional efficiency, depends on concentration of nutrients outside cell
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency

        dadt = k_n * (self.phiR_max - phi_R - phi_S) - k_t * (phi_R - self.phiR_min)
        return dadt

    def dUdt(self, U, phi_S, phi_R, a, b):
        # damage concentration ODE
        return self.alpha*phi_R*b - self.beta*phi_S*U - U*self.GrowthRate(a, phi_R, U)

    def dXdt(self, X, a, phi_R, V, U):
        # division protein ODE
        dxdt = self.f_X(a,U) * self.GrowthRate(a, phi_R, U) * V - self.mu * X
        return dxdt

    def dVdt(self, V, a, phi_R, U):
        # cell volume ODE
        dvdt = self.GrowthRate(a, phi_R, U) * V
        return dvdt
    
    # initialize simulation
    def phiR_ss(self, a, U, k_n0):
        # function for phi_R at steady state
        k_n = k_n0 * self.f(a) # nutritional efficiency, depends on concentration of nutrients outside cell
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency
        return (k_n*(self.phiR_max-self.f_S(U)) + k_t*self.phiR_min) / (k_n + k_t)
    
    def func(self, x, k_n0, b):
        # function for calculating steady state conditions for given parameters
        return [self.phiR_ss(x[0],x[2],k_n0) - x[1], # x[0]=a, x[1]=phi_R, x[2]=U
                self.f_R(x[0],x[2]) - x[1],
                self.beta*x[2]*self.f_S(x[2]) - self.alpha*x[1]*b]
    
    def initialize(self, num_cells_init, k_n0, b):
        self._log = []
        # self.k_n0 = np.random.normal(loc=self.k_n0_mean, scale=0.2*self.k_n0_mean)
        self.true_num_cells = num_cells_init

        # solving for initial conditions to produce steady state
        a0, phi_R0, U0, phi_S0 = self._guess_init_values(k_n0, b)

        # initializing each array to save initial conditions for each new trajectory
        phiR_birth = np.ones((num_cells_init))*phi_R0
        phiS_birth = np.ones((num_cells_init))*phi_S0
        a_birth = np.ones((num_cells_init))*a0
        U_birth = np.ones((num_cells_init))*U0
        # assigning random initial cell volume, in um^3
        cycle_t = np.log(2) / self.GrowthRate(a0, phi_R0, U0)

        start_t_stack = np.random.randint(int(cycle_t), int(cycle_t*100) + 1, (num_cells_init))/100
        birth_size = 1 / self.f_X(a0, U0) # average cell size at birth at initial steady-state growth
        V_birth = birth_size * np.exp(self.GrowthRate(a0, phi_R0, U0) * start_t_stack)
        X_birth = self.f_X(a0,U0) * V_birth * 0.5

        self._t = 0
        self._log.append([self._t, k_n0, b, num_cells_init, U0])
        self.populations = np.stack((phiR_birth, phiS_birth, a_birth, U_birth, X_birth, V_birth), axis=0)

    def _guess_init_values(self, k_n0, b):
        while True:
            if self.guess_value_list is not None:
                guess_value_list = self.guess_value_list
                a0,phi_R0,U0 = optimize.fsolve(self.func, guess_value_list, args=(k_n0, b))
            else:
                # 1e-5 to 1e-4
                # 0.2 to 0.5
                # 1e-4 to 1e-2
                value1 = np.random.uniform(1e-5, 1e-4)
                value2 = np.random.uniform(0.2, 0.5)
                value3 = np.random.uniform(1e-4, 1e-2)
                guess_value_list = [value1, value2, value3]
                a0,phi_R0,U0 = optimize.fsolve(self.func, guess_value_list, args=(k_n0, b)) # requires guess of initial conditions

            phi_S0 = self.f_S(U0)
            ls = [a0, phi_R0, U0, phi_S0]
            if all(val >= 0 for val in ls):
                self.guess_value_list = guess_value_list
                # print(ls)
                print("Initial values are physical now")
                break
            else:
                self.guess_value_list = None
                print(ls)
                print("Initial values are unphysical, changing guess of initial conditions")
        return ls
        
    # simulatation implementation
    def MultiIntegrate(self, Species, dt, b, k_n0):
        # numerically solve via Euler-Maruyama method
        phiR_i,phiS_i,a_i,U_i,X_i,V_i = Species

        phi_R = phiR_i + self.dphiR_dt(phiR_i, a_i, U_i)*dt
        phi_S = phiS_i + self.dphiS_dt(phiS_i, a_i, phiR_i, U_i)*dt
        a = a_i + self.dAAdt(a_i, phiR_i, phiS_i, U_i, k_n0)*dt
        # ensure that amino acid conc. is not negative
        a[a < 1e-7] = 1e-7
        # if a < 1e-7:
        #     print('Warning: amino acid conc. went negative and was reset, consider decreasing integration step size')
        #     a = 1e-7

        # adding noise to U
        noise = np.sqrt(2*self.sigma) * np.sqrt(dt) * np.random.normal(size=U_i.shape)
        U = U_i + self.dUdt(U_i, phiS_i, phiR_i, a_i, b)*dt + noise
        # check to make sure value is physical
        if b == 0:
            U[...] = 0
        U[U < 0] = 0
        # if U < 0 or b == 0:
        #     U = 0

        X = X_i + self.dXdt(X_i, a_i, phiR_i, V_i, U_i)*dt
        V = V_i + self.dVdt(V_i, a_i, phiR_i, U_i)*dt

        return np.stack((phi_R, phi_S, a, U, X, V), axis=0)

    def _cell_population_truncate(self, threshold):
        num_entity = self.populations.shape[0]
        num_cells_saved = self.populations.shape[-1]

        if num_cells_saved > threshold:
            # downsampling if population exceeds threshold
            row_ids = np.random.randint(num_cells_saved, size = threshold)
            self.populations = self.populations[:,row_ids]
            num_cells = threshold
        elif (num_cells_saved < threshold) and (self.true_num_cells > num_cells_saved):
            # upsampling if population decreased, but is still above number currently being simulated
            num_cells_add = int(np.min((threshold-num_cells_saved, self.true_num_cells-num_cells_saved)))
            num_cells = int(num_cells_saved+num_cells_add)
            # t_birth = np.ones((num_cells,1))*t_birth[0]

            population_new = np.random.normal(size=(num_entity,num_cells_add))
            population_new = population_new * self.populations.std(axis=-1, keepdims=True) \
                + self.populations.mean(axis=-1, keepdims=True)
            
            population_new = population_new.clip(0, 0.99)
            population_new[0] = population_new[0].clip(self.phiR_min, self.phiR_max)
            population_new[1] = population_new[1].clip(None, self.phiS_max)
            population_new[2] = population_new[2].clip(1e-7, None)
            population_new[5] = population_new[5].clip(self.populations[5].min(), self.populations[5].max())

            self.populations = np.concatenate((self.populations, population_new), -1)
        else:
            num_cells = num_cells_saved

        return num_cells
    
    def simulate_population(self, k_n0, b, delta_t, n_steps=3000, threshold=50):
        if np.isnan(self.populations).any() or (self.populations < 0).any(): # checking to make sure nan values are not present
            # print('populations:', self.populations)
            raise ValueError(f'Simulation error, nan or negative values present')
        
        num_cells = self._cell_population_truncate(threshold)
        # unpacking initial conditions for each cell trajectory

        iterations = int(delta_t * n_steps)
        dt = 1 / n_steps
        cell_count = [num_cells]
        
        if isinstance(k_n0, float) or len(k_n0) == 1:
            k_n0_list = np.ones(iterations) * k_n0
        else:
            k_n0_list = k_n0

        species_stack = self.populations
        for i in range(iterations):
            species_stack = self.MultiIntegrate(species_stack, dt, b, k_n0_list[i]) # integrating one timestep

            X_0 = 1 # amount of division proteins required to trigger division
            # if cell has added threshold volume amount, it will then divide
            birth_check = species_stack[4] >= X_0
            if birth_check.sum() != 0:
                r = np.random.normal(0.5, 0.04, birth_check.sum())

                X_stack_children = species_stack[:,birth_check].copy()
                X_stack_children[4] = 0
                X_stack_children[5] = X_stack_children[5] * (1 - r)

                species_stack[4,birth_check] = 0
                species_stack[5,birth_check] = species_stack[5,birth_check] * r

                species_stack = np.concatenate((species_stack, X_stack_children), -1)

            # if cell has accumulated sufficient damage, it will die
            death_check = species_stack[3] >= 1
            if death_check.sum() != 0:
                species_stack = species_stack[:,~death_check]

            cell_count.append(species_stack.shape[-1])
            if cell_count[-1] == 0:
                break

        if self.true_num_cells > threshold:
            true_num_cells_next = np.round(self.true_num_cells * (cell_count[-1] / num_cells))
        else:
            true_num_cells_next = cell_count[-1]

        true_num_cells_prev = self.true_num_cells
        self.true_num_cells = true_num_cells_next

        self.populations = species_stack
        self._t = self._t + delta_t
        U_ave = species_stack[3].mean() if species_stack[3].size > 0 else 1
        self._log.append([self._t, k_n0_list[-1], b, self.true_num_cells, U_ave])
        return self._t, (true_num_cells_prev, true_num_cells_next)
        # return self._t, self.true_num_cells

    @property
    def logger(self):
        return self._log

    @property
    def time_point(self):
        return self._t


# class Scenario(object):
#     pass