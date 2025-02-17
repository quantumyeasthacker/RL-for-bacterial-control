import numpy as np
from scipy import integrate, signal, optimize
import random


class Cell_Population:
    def __init__(self, num_cells_init, delta_t, omega, kn0_mean, T):
        self.init_conditions = []
        self.num_cells_init = num_cells_init
        self.delta_t = delta_t
        self.omega = omega

        # parameters
        self.phiR_min = 0
        self.phiR_max = 0.55 # Scott et al. 2010
        self.a_n = 1e-3 # level to trigger negative feeback inhibition of amino acid supply, Scott et al. 2014
        self.a_t = 1e-4 # amino acid level for efficient peptide elongation, Scott et al. 2014
        self.n_f = 2 # cooperativity in feedback
        self.n_g = 2
        # Kratz and Banerjee 2023
        self.sigma = 0.015 # noise strength
        self.alphaX = 4.5
        self.betaX = 1.1
        self.mu = 0.6
        self.kt0_mean = 2.7 # translational efficiency
        self.q = 2
        self.phiSm_mean = 0.33
        self.alpha = 1.54
        self.beta = 10.5
        self.K_u = 0.076

        self.kn0_mean = kn0_mean
        self.sigma_kn0 = 0
        self.T = T


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
        return np.maximum(ut, 0)

    def GrowthRate(self, a, phi_R, U):
        # growth rate function
        k_t = self.k_t0 * self.g(a) * self.U_t(U)
        k = k_t * (phi_R - self.phiR_min)
        return k

    def dphiR_dt(self, phi_R, t, a, U):
        # ribosome mass fraction ODE
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency
        dpdt = k_t * (phi_R - self.phiR_min) * (self.f_R(a,U) - phi_R)
        return dpdt

    def dphiS_dt(self, phi_S, t, a, phi_R, U):
        # stress sector mass fraction ODE
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency
        dpdt = k_t * (phi_R - self.phiR_min) * (self.f_S(U) - phi_S)
        return dpdt

    def dAAdt(self, a, t, phi_R, phi_S, U, k_n0):
        # amino acid concentration ODE (variable nutrient conc.(c))
        k_n = k_n0 * self.f(a) # nutritional efficiency, depends on concentration of nutrients outside cell
        k_t = self.k_t0 * self.g(a) * self.U_t(U) # translational efficiency

        dadt = k_n * (self.phiR_max - phi_R - phi_S) - k_t * (phi_R - self.phiR_min)
        return dadt

    def dUdt(self, U, t, phi_S, phi_R, a, b):
        # damage concentration ODE
        return self.alpha*phi_R*b - self.beta*phi_S*U - U*self.GrowthRate(a, phi_R, U)

    def dXdt(self, X, t, a, phi_R, V, U):
        # division protein ODE
        dxdt = self.f_X(a,U) * self.GrowthRate(a, phi_R, U) * V - self.mu * X
        return dxdt

    def dVdt(self, V, t, a, phi_R, U):
        # cell volume ODE
        dvdt = self.GrowthRate(a, phi_R, U) * V
        return dvdt

    def dkn0dt(self, t, k_n0):
        tau = 3
        Amp = 2

        drift = Amp*np.sin(t*2*np.pi/self.T + self.phase) + self.kn0_mean
        dkdt = -(1/tau)*(k_n0 - drift)
        return dkdt

    def MultiIntegrate(self, Species, t, dt, b, k_n0):
        # numerically solve via Euler-Maruyama method
        phiR_i,phiS_i,a_i,U_i,X_i,V_i = Species

        phi_R = phiR_i + self.dphiR_dt(phiR_i, t, a_i, U_i)*dt
        phi_S = phiS_i + self.dphiS_dt(phiS_i, t, a_i, phiR_i, U_i)*dt
        a = a_i + self.dAAdt(a_i, t, phiR_i, phiS_i, U_i, k_n0)*dt
        # ensure that amino acid conc. is not negative
        a[a < 1e-7] = 1e-7
        #   print('Warning: amino acid conc. went negative and was reset, consider decreasing integration step size')

        # adding noise to U
        noise = np.sqrt(2*self.sigma) * np.sqrt(dt) * np.random.normal(size=U_i.shape)
        U = U_i + self.dUdt(U_i, t, phiS_i, phiR_i, a_i, b)*dt + noise
        # check to make sure value is physical
        if b == 0:
            U[:] = 0
        U[U < 0] = 0

        X = X_i + self.dXdt(X_i, t, a_i, phiR_i, V_i, U_i)*dt
        V = V_i + self.dVdt(V_i, t, a_i, phiR_i, U_i)*dt

        return np.array([phi_R,phi_S,a,U,X,V])

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
    
    def func_0(self, x, k_n0):
        # function for calculating steady state conditions for given parameters, used if b=0 to reduce complexity
        return [self.phiR_ss(x[0],0,k_n0) - x[1], # x[0]=a, x[1]=phi_R
                self.f_R(x[0],0) - x[1]]

    def initialize(self, b, rand_param=False):
        # self.k_n0 = np.random.normal(loc=self.kn0_mean, scale=0.2*self.kn0_mean)
        self.k_n0 = self.kn0_mean
        self.phase = np.random.uniform(0,self.T)

        if rand_param:
            k_t0 = np.random.normal(loc=self.kt0_mean, scale=0.1*self.kt0_mean)
            self.k_t0 = np.clip(k_t0, 1.5,4)
            phiS_max = np.random.normal(loc=self.phiSm_mean, scale=0.1*self.phiSm_mean)
            self.phiS_max = np.clip(phiS_max, 0.1,0.5)
        else:
            self.k_t0 = self.kt0_mean
            self.phiS_max = self.phiSm_mean

        # solving for initial conditions to produce steady state
        if b > 0:
            a0,phi_R0,U0 = optimize.fsolve(self.func, [1e-4, 0.3, 1e-3], args=(self.k_n0,b)) # requires guess of initial conditions
            phi_S0 = self.f_S(U0)
        elif b == 0:
            a0,phi_R0 = optimize.fsolve(self.func_0, [1e-4, 0.3], args=(self.k_n0))
            U0 = 0.0
            phi_S0 = 0.0
        ls = [a0,phi_R0,U0,phi_S0]
        if not all(val >= 0 for val in ls):
            print(ls)
            raise ValueError(f'Initial values are unphysical, change guess of initial conditions')

        # initializing each array to save initial conditions for each new trajectory
        phiR_birth = np.ones((self.num_cells_init))*phi_R0
        phiS_birth = np.ones((self.num_cells_init))*phi_S0
        a_birth = np.ones((self.num_cells_init))*a0
        U_birth = np.ones((self.num_cells_init))*U0
        # assigning random initial cell volume, in um^3
        V_birth = np.zeros((self.num_cells_init))
        X_birth = np.zeros((self.num_cells_init))
        cycle_t = np.log(2) / self.GrowthRate(a0, phi_R0, U0)
        for i in range(self.num_cells_init):

            start_t = random.randint(int(cycle_t),int(cycle_t*100))/100 # assigning random initial cell volume, in um^3
            birth_size = 1 / self.f_X(a0, U0) # average cell size at birth at initial steady-state growth
            V_birth[i] = birth_size * np.exp(self.GrowthRate(a0, phi_R0, U0) * start_t)

            X_birth[i] = self.f_X(a0,U0)*V_birth[i]*0.5

        self.init_conditions = np.array([phiR_birth, phiS_birth, a_birth, U_birth, X_birth, V_birth])
        self.t_start = 0
        self.t_stop = self.delta_t

    def upsample(self, val_array, num_cells_add, clip_high=0.99, clip_low=0):
        samples = np.random.normal(np.mean(val_array), np.std(val_array), num_cells_add)
        val_array = np.concatenate((val_array, samples), -1)
        return val_array.clip(clip_low, clip_high)

    # simulatation implementation

    def simulate_population(self, true_num_cells, b, n_steps=3000, threshold=50):

        # unpacking initial conditions for each cell trajectory
        phiR_birth, phiS_birth, a_birth, U_birth, X_birth, V_birth = self.init_conditions.copy()
        if any(np.isnan(a_birth)) or any(a_birth < 0): # checking to make sure nan values are not present
            print('a_start:',a_birth)
            raise ValueError(f'Simulation error, nan or negative values present')
        num_cells_saved = len(V_birth)

        if num_cells_saved > threshold:
            # downsampling if population exceeds threshold
            row_ids = random.sample(range(num_cells_saved), threshold)
            phiR_birth = phiR_birth[row_ids]
            phiS_birth = phiS_birth[row_ids]
            a_birth = a_birth[row_ids]
            U_birth = U_birth[row_ids]
            X_birth = X_birth[row_ids]
            V_birth = V_birth[row_ids]
            num_cells = threshold
        elif (num_cells_saved < threshold) & (true_num_cells > num_cells_saved):
            # upsampling if population decreased, but is still above number currently being simulated
            num_cells_add = int(np.min((threshold-num_cells_saved, true_num_cells-num_cells_saved)))
            num_cells = int(num_cells_saved+num_cells_add)
            phiR_birth = self.upsample(phiR_birth, num_cells_add, self.phiR_max, self.phiR_min)
            phiS_birth = self.upsample(phiS_birth, num_cells_add, self.phiS_max)
            a_birth = self.upsample(a_birth, num_cells_add, clip_low=1e-7)
            U_birth = self.upsample(U_birth, num_cells_add)
            X_birth = self.upsample(X_birth, num_cells_add)
            V_birth = self.upsample(V_birth, num_cells_add, np.max(V_birth), np.min(V_birth))
        else:
            num_cells = num_cells_saved

        iterations = int((self.t_stop - self.t_start)*n_steps)
        t = np.linspace(self.t_start,self.t_stop,iterations)
        dt = (self.t_stop - self.t_start)/iterations
        cell_count = [num_cells]

        # first simulate nutrient environment
        k_n0 = np.zeros(iterations)
        k_n0[0] = self.k_n0
        for i in range(1,iterations):
            k_n0[i] = k_n0[i-1] + self.dkn0dt(t[i-1], k_n0[i-1])*dt + np.sqrt(2*self.sigma_kn0)*np.sqrt(dt)*np.random.normal()
        k_n0 = np.clip(k_n0, 0.1, 5.0) # clipping values to keep in physiological range
        self.k_n0 = k_n0[-1]

        species_stack = np.array([phiR_birth, phiS_birth, a_birth, U_birth, X_birth, V_birth])
        for i in range(1, iterations):
            species_stack = self.MultiIntegrate(species_stack, t[i], dt, b, k_n0[i-1]) # integrating one timestep

            X_0 = 1 # amount of division proteins required to trigger division
            # if cell has added threshold volume amount, it will then divide
            birth_check = species_stack[4,:] >= X_0
            if birth_check.sum() != 0:
                r = np.random.normal(0.5, 0.04, birth_check.sum())

                X_stack_children = species_stack[:,birth_check].copy()
                X_stack_children[4,:] = 0
                X_stack_children[5,:] = X_stack_children[5,:] * (1 - r)

                species_stack[4,birth_check] = 0
                species_stack[5,birth_check] = species_stack[5,birth_check] * r

                species_stack = np.concatenate((species_stack, X_stack_children), -1)

            # if cell has accumulated sufficient damage, it will die
            death_check = species_stack[3,:] >= 1
            if death_check.sum() != 0:
                species_stack = species_stack[:,~death_check]

            cell_count.append(species_stack.shape[1])
            if cell_count[-1] == 0:
                break

        self.init_conditions = species_stack.copy()
        self.t_start = self.t_stop
        self.t_stop += self.delta_t

        if true_num_cells > threshold:
            true_num_cells_next = np.round(true_num_cells * (cell_count[-1] / num_cells))
        else:
            true_num_cells_next = cell_count[-1]

        return t, [true_num_cells, true_num_cells_next]


    def get_reward_all(self, state, cell_count, b):
        # separating state vector
        state_gr, state_kn0, state_act = np.array_split(state, 3)
        state_gr = state_gr.tolist()
        state_kn0 = state_kn0.tolist()
        state_act = state_act.tolist()

        p_init = cell_count[0]
        p_final = cell_count[-1]

        if p_final == 0:
            growth_rate = -10
        else:
            growth_rate = (np.log(p_final) - np.log(p_init)) / self.delta_t
        # dropping oldest values and replacing with newest ones
        state_gr.pop(0) 
        state_gr.append(growth_rate)
        state_kn0.pop(0)
        state_kn0.append(self.k_n0)
        state_act.pop(0)
        state_act.append(b)

        state = state_gr + state_kn0 + state_act
        cost = growth_rate + self.omega*b**2 # adding nonlinear penalty for drug application

        return [state, cost]

    def get_reward_no_antibiotic(self, state, cell_count, b):
        # separating state vector
        state_gr, state_kn0 = np.array_split(state, 2)
        state_gr = state_gr.tolist()
        state_kn0 = state_kn0.tolist()

        p_init = cell_count[0]
        p_final = cell_count[-1]

        if p_final == 0:
            growth_rate = -10
        else:
            growth_rate = (np.log(p_final) - np.log(p_init)) / self.delta_t
        # dropping oldest values and replacing with newest ones
        state_gr.pop(0) 
        state_gr.append(growth_rate)
        state_kn0.pop(0)
        state_kn0.append(self.k_n0)

        state = state_gr + state_kn0
        cost = growth_rate + self.omega*b**2 # adding nonlinear penalty for drug application

        return [state, cost]

    def get_reward_no_nutrient(self, state, cell_count, b):
        # separating state vector
        state_gr, state_act = np.array_split(state, 2)
        state_gr = state_gr.tolist()
        state_act = state_act.tolist()

        p_init = cell_count[0]
        p_final = cell_count[-1]

        if p_final == 0:
            growth_rate = -10
        else:
            growth_rate = (np.log(p_final) - np.log(p_init)) / self.delta_t
        # dropping oldest values and replacing with newest ones
        state_gr.pop(0) 
        state_gr.append(growth_rate)
        state_act.pop(0)
        state_act.append(b)

        state = state_gr + state_act
        cost = growth_rate + self.omega*b**2 # adding nonlinear penalty for drug application

        return [state, cost]

if __name__ == '__main__':
    pass
#     sim_controller = Cell_Population(60, 0.2, 0.02, 2.55, 12)
#     sim_controller.initialize(0.0)
#     t, cell_count = sim_controller.simulate_population(sim_controller.num_cells_init, 4.0)
#     t, cell_count = sim_controller.simulate_population(sim_controller.num_cells_init, 4.0)