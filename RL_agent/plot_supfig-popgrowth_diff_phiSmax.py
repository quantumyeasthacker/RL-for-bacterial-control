# %%
from envs.cell_model import CellConfig, Cell_Population
import numpy as np
import matplotlib.pyplot as plt

# %%
int_steps = 10000
delta_t = 1
iterations = 75
kn0_list = np.ones(int_steps)*4

# %%
CellConfig.phiS_max = 0.33 # default
cell_pop = Cell_Population(CellConfig)
cell_pop.initialize(100, 4, 0)


t = np.zeros(iterations)
cells = np.zeros(iterations)

b = 0
t[1], cell = cell_pop.simulate_population(kn0_list, b, delta_t)
cells[:2] = cell
for i in range(2,iterations):
    if b == 0:
        b = 3.72
    elif b == 3.72:
        b = 0
    t[i], cell = cell_pop.simulate_population(kn0_list, b, delta_t)
    cells[i] = cell[1]

# %%
CellConfig.phiS_max = 0.33*0.9
cell_pop = Cell_Population(CellConfig)
cell_pop.initialize(100, 4, 0)


t_low = np.zeros(iterations)
cells_low = np.zeros(iterations)

b = 0
t_low[1], cell = cell_pop.simulate_population(kn0_list, b, delta_t)
cells_low[:2] = cell
for i in range(2,iterations):
    if b == 0:
        b = 3.72
    elif b == 3.72:
        b = 0
    t_low[i], cell = cell_pop.simulate_population(kn0_list, b, delta_t)
    cells_low[i] = cell[1]

# %%
CellConfig.phiS_max = 0.33*1.1
cell_pop = Cell_Population(CellConfig)
cell_pop.initialize(100, 4, 0)


t_high = np.zeros(iterations)
cells_high = np.zeros(iterations)

b = 0
t_high[1], cell = cell_pop.simulate_population(kn0_list, b, delta_t)
cells_high[:2] = cell
for i in range(2,iterations):
    if b == 0:
        b = 3.72
    elif b == 3.72:
        b = 0
    t_high[i], cell = cell_pop.simulate_population(kn0_list, b, delta_t)
    cells_high[i] = cell[1]

# %%
plt.rcParams.update({'font.size': 16})

fig = plt.figure()
plt.plot(t_high,cells_high, label='high ($1.1 \\times \phi_S^\max$)')
plt.plot(t,cells, label='normal ($\phi_S^\max$)')
plt.plot(t_low,cells_low, label='low ($0.9 \\times \phi_S^\max$)')
plt.xlabel('Time (h)')
plt.ylabel('Population Size')
plt.yscale('log')
plt.legend(fontsize=14)
plt.show()
fig.savefig("supfig-popgrowth_diff_phiSmax.pdf", dpi=300, bbox_inches='tight')