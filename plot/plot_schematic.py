# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %%
with open("/mnt/c/Users/zhwen/Dropbox/BacteriaAdaptation/20250325_varenv/results_delay_30_varenv_eval/a3.72_T6_delay30_rep0/trial_1tcbk.pkl", "rb") as f:
    info = pickle.load(f)
    tkbc = np.array(info["log"])
    tcbk = tkbc[:,[0,3,2,1,5,6]].T

t = tcbk[0]
b = tcbk[2]
k = tcbk[3]

# %% constant
# x = np.linspace(0, 10, 100)
# y = np.ones_like(x) * 5
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, k,)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Time")
ax.set_ylabel("Nutrient")
ax.set_xlim(0, None)
ax.set_ylim(0, np.max(k))

# ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)

ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

fig.savefig("fig_schematic/flcut_nut.jpg", dpi = 600, bbox_inches='tight')

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, 3 * np.ones_like(k))

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Time")
ax.set_ylabel("Nutrient")
ax.set_xlim(0, None)
ax.set_ylim(0, np.max(k))

# ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)

ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

fig.savefig("fig_schematic/flcut_constant_3.jpg", dpi = 600, bbox_inches='tight')

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, 1 * np.ones_like(k))

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Time")
ax.set_ylabel("Nutrient")
ax.set_xlim(0, None)
ax.set_ylim(0, np.max(k))

# ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)

ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

fig.savefig("fig_schematic/flcut_constant_1.jpg", dpi = 600, bbox_inches='tight')


# %%
