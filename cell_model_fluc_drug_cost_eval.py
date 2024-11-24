import numpy as np
from scipy import integrate, signal, optimize
import random


class Cell_Population:
    def __init__(self, num_cells_init, delta_t, omega, kn0_mean, T):
        self.init_conditions = []
        self.t_start = []
        self.t_stop = []
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
        self.k_t0 = 2.7 # translational efficiency
        self.q = 2
        self.phiS_max = 0.33
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
        return max(0, ut)

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
        phase = 0

        drift = Amp*np.sin(t*2*np.pi/self.T + phase) + self.kn0_mean
        dkdt = -(1/tau)*(k_n0 - drift)
        return dkdt


    def integrate(self, Species, t, dt, b, k_n0):
        # numerically solve via Euler-Maruyama method
        phiR_i,phiS_i,a_i,U_i,X_i,V_i = Species

        phi_R = phiR_i + self.dphiR_dt(phiR_i, t, a_i, U_i)*dt
        phi_S = phiS_i + self.dphiS_dt(phiS_i, t, a_i, phiR_i, U_i)*dt
        a = a_i + self.dAAdt(a_i, t, phiR_i, phiS_i, U_i, k_n0)*dt
        # ensure that amino acid conc. is not negative
        if a < 1e-7:
            print('Warning: amino acid conc. went negative and was reset, consider decreasing integration step size')
            a = 1e-7

        # adding noise to U
        noise = np.sqrt(2*self.sigma) * np.sqrt(dt) * np.random.normal()
        U = U_i + self.dUdt(U_i, t, phiS_i, phiR_i, a_i, b)*dt + noise
        # check to make sure value is physical
        if U < 0 or b == 0:
            U = 0

        X = X_i + self.dXdt(X_i, t, a_i, phiR_i, V_i, U_i)*dt
        V = V_i + self.dVdt(V_i, t, a_i, phiR_i, U_i)*dt

        return phi_R,phi_S,a,U,X,V


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

    def initialize(self, b):
        # self.k_n0 = np.random.normal(loc=self.kn0_mean, scale=0.2*self.kn0_mean)
        self.k_n0 = self.kn0_mean

        # solving for initial conditions to produce steady state
        a0,phi_R0,U0 = optimize.fsolve(self.func, [1e-4, 0.3, 1e-3], args=(self.k_n0,b)) # requires guess of initial conditions
        phi_S0 = self.f_S(U0)
        ls = [a0,phi_R0,U0,phi_S0]
        if not all(val >= 0 for val in ls):
            print(ls)
            raise ValueError(f'Initial values are unphysical, change guess of initial conditions')

        # initializing each array to save initial conditions for each new trajectory
        t_birth = np.zeros((self.num_cells_init,1))
        phiR_birth = np.ones((self.num_cells_init,1))*phi_R0
        phiS_birth = np.ones((self.num_cells_init,1))*phi_S0
        a_birth = np.ones((self.num_cells_init,1))*a0
        U_birth = np.ones((self.num_cells_init,1))*U0
        # assigning random initial cell volume, in um^3
        V_birth = np.zeros((self.num_cells_init,1))
        X_birth = np.zeros((self.num_cells_init,1))
        cycle_t = np.log(2) / self.GrowthRate(a0, phi_R0, U0)
        for i in range(self.num_cells_init):

            start_t = random.randint(int(cycle_t),int(cycle_t*100))/100 # assigning random initial cell volume, in um^3
            birth_size = 1 / self.f_X(a0, U0) # average cell size at birth at initial steady-state growth
            V_birth[i] = birth_size * np.exp(self.GrowthRate(a0, phi_R0, U0) * start_t)

            X_birth[i] = self.f_X(a0,U0)*V_birth[i]*0.5

        self.init_conditions = (t_birth, phiR_birth, phiS_birth, a_birth, U_birth, X_birth, V_birth)
        self.t_start = 0
        self.t_stop = self.delta_t

    def upsample(self, val_array, num_cells_add, clip_high=0.99, clip_low=0):
        samples = np.random.normal(np.mean(val_array), np.std(val_array), num_cells_add)
        val_array = np.concatenate((val_array,samples.reshape(len(samples),1)))
        return val_array.clip(clip_low, clip_high)

    # simulatation implementation

    def simulate_population(self, true_num_cells, b, n_steps=3000, threshold=50):

        # unpacking initial conditions for each cell trajectory
        t_birth, phiR_birth, phiS_birth, a_birth, U_birth, X_birth, V_birth = self.init_conditions
        if any(np.isnan(a_birth)) or any(a_birth < 0): # checking to make sure nan values are not present
            print('a_start:',a_birth)
            raise ValueError(f'Simulation error, nan or negative values present')
        num_cells_saved = len(t_birth)

        if num_cells_saved > threshold:
            # downsampling if population exceeds threshold
            row_ids = random.sample(range(num_cells_saved-1), threshold)
            t_birth = t_birth[row_ids]
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
            t_birth = np.ones((threshold,1))*t_birth[0]
            phiR_birth = self.upsample(phiR_birth, num_cells_add, self.phiR_max, self.phiR_min)
            phiS_birth = self.upsample(phiS_birth, num_cells_add, self.phiS_max)
            a_birth = self.upsample(a_birth, num_cells_add, clip_low=1e-7)
            U_birth = self.upsample(U_birth, num_cells_add)
            X_birth = self.upsample(X_birth, num_cells_add)
            V_birth = self.upsample(V_birth, num_cells_add, np.max(V_birth), np.min(V_birth))
            num_cells = threshold
        else:
            num_cells = num_cells_saved

        iterations = int((self.t_stop - self.t_start)*n_steps)
        t = np.linspace(self.t_start,self.t_stop,iterations)
        dt = (self.t_stop - self.t_start)/iterations
        cell_count = np.ones(iterations) * num_cells
        cell_birth = np.zeros(iterations)
        cell_death = np.zeros(iterations)

        t_end = np.zeros_like(t_birth)
        V_end = np.zeros_like(V_birth)
        a_end = np.zeros_like(a_birth)
        phiR_end = np.zeros_like(phiR_birth)
        phiS_end = np.zeros_like(phiS_birth)
        U_end = np.zeros_like(U_birth)
        X_end = np.zeros_like(X_birth)

        # first simulate nutrient environment
        k_n0 = np.zeros(iterations)
        k_n0[0] = self.k_n0
        for i in range(1,iterations):
            k_n0[i] = k_n0[i-1] + self.dkn0dt(t[i-1], k_n0[i-1])*dt + np.sqrt(2*self.sigma_kn0)*np.sqrt(dt)*np.random.normal()
        k_n0 = np.clip(k_n0, 0.1, 5.0) # clipping values to keep in physiological range
        self.k_n0 = k_n0[-1]

        start = True
        m=0
        while m < len(t_birth):
            # start at i=1 so a[i-1]=a0 and phi_R[i-1]=phi_R0
            i = np.where(t == t_birth[m])[0][0] + 1 # starting at time index corresponding to birth a cell lineage

            V = np.zeros(iterations)
            V[i-1] = V_birth[m]

            phi_R = np.zeros(iterations)
            phi_R[i-1] = phiR_birth[m]

            phi_S = np.zeros(iterations)
            phi_S[i-1] = phiS_birth[m]

            a = np.zeros(iterations)
            a[i-1] = a_birth[m]

            U = np.zeros(iterations)
            U[i-1] = U_birth[m]

            X = np.zeros(iterations)
            X[i-1] = X_birth[m]

            while i < iterations:

                species_0 = [phi_R[i-1], phi_S[i-1], a[i-1], U[i-1], X[i-1], V[i-1]] # packing initial conditions

                phi_R[i], phi_S[i], a[i], U[i], X[i], V[i] = self.integrate(species_0, t[i-1], dt, b, k_n0[i-1]) # integrating one timestep

                X_0 = 1 # amount of division proteins required to trigger division
                # if cell has added threshold volume amount, it will then divide
                if X[i-1] >= X_0:

                    r = np.random.normal(0.5, 0.04) # drawing random value for volume allocation to daughter cell, taken from normal distribution
                    V[i] = r * V[i-1] # cell volume is divided roughly in half

                    # updating initial conditions arrays
                    t_birth = np.vstack((t_birth, np.array(t[i])))
                    V_birth = np.vstack((V_birth, np.array( (1 - r) * V[i-1] )))
                    a_birth = np.vstack((a_birth, np.array(a[i])))
                    phiR_birth = np.vstack((phiR_birth, np.array(phi_R[i])))
                    phiS_birth = np.vstack((phiS_birth, np.array(phi_S[i])))
                    U_birth = np.vstack((U_birth, np.array(U[i])))
                    X_birth = np.vstack((X_birth, np.array(0))) # division protein concentration is reset to zero

                    cell_count[i:iterations] +=1 # keeping track of total cell count over time
                    cell_birth[i] +=1 # recording birth dynamics

                # if cell has accumulated sufficient damage, it will die
                if U[i-1] >= 1:
                    cell_count[i:iterations] -=1 # keeping track of total cell count over time
                    cell_death[i] +=1 # recording death dynamics
                    i = iterations # if cells dies, exits while loop and simulates next cell

                # saving simulation results for next call
                if (i == iterations-1) and not start:
                    t_end = np.vstack((t_end, np.array(t[i])))
                    V_end = np.vstack((V_end, np.array(V[i])))
                    a_end = np.vstack((a_end, np.array(a[i])))
                    phiR_end = np.vstack((phiR_end, np.array(phi_R[i])))
                    phiS_end = np.vstack((phiS_end, np.array(phi_S[i])))
                    U_end = np.vstack((U_end, np.array(U[i])))
                    X_end = np.vstack((X_end, np.array(X[i])))

                if (i == iterations-1) and start:
                    t_end = np.array(t[i])
                    V_end = np.array(V[i])
                    a_end = np.array(a[i])
                    phiR_end = np.array(phi_R[i])
                    phiS_end = np.array(phi_S[i])
                    U_end = np.array(U[i])
                    X_end = np.array(X[i])
                    start = False

                i +=1
            m +=1

        final_conditions = (t_end, phiR_end, phiS_end, a_end, U_end, X_end, V_end)
        self.init_conditions = final_conditions
        self.t_start = t[-1]
        self.t_stop = self.t_start + self.delta_t
        try: len(t_end)
        except: cell_count = np.zeros(cell_count.shape)

        if true_num_cells > threshold:
            true_num_cells_next = np.round(true_num_cells * (1 + (cell_count[-1] - num_cells) / num_cells))
        else:
            true_num_cells_next = cell_count[-1]

        return t, [true_num_cells, true_num_cells_next]


    def get_state_reward(self, state, cell_count, b):
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