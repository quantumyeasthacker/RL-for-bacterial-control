import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy import signal


# default_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# default_color_list = ["#dec60c", "#a7c82f", "#548c6a"]
COLOR_LIST = ["#dec60c", "#a7c82f", "#548c6a"]

ANTIBIOTIC_COLOR = "#216d87"
# nutrient color ["#dec60c", "#a7c82f", "#548c6a"]
POP_SIZE_COLOR = "#35a97b"


def expand_and_fill(vec, threshold):
    """Expands the vector to the given threshold length, filling with zeros if necessary."""
    if len(vec) >= threshold:
        return vec[:threshold]  # Trim if longer
    else:
        return np.pad(vec, (0, threshold - len(vec)), mode='constant', constant_values=0)


def estimate_frequency_fft(sequence, axis=-1, sampling_unit=1.0):
    edge_idx = down_edge_detection(sequence, axis=axis)
    if np.sum(edge_idx) == 0:
        return np.nan
    
    n = np.shape(sequence)[axis]
    fft_vals = np.abs(np.fft.rfft(sequence, axis=axis))  # Compute magnitude of FFT
    freqs = np.fft.rfftfreq(n, d=sampling_unit)  # Frequency values

    peak_idx = np.argmax(fft_vals[1:], axis=axis) + 1  # Ignore DC component (index 0)
    estimated_freq = freqs[peak_idx]
    # estimated_period = 1 / estimated_freq if estimated_freq != 0 else np.inf

    return estimated_freq

def down_edge_detection(sequence, axis=-1):
    n = np.shape(sequence)[axis]
    edge_idx = np.zeros(n, dtype=bool)
    for i in range(1, n):
        edge_idx[i] = sequence[i] < sequence[i - 1]
    return edge_idx


def cross_correlation(sig1, sig2):
    n_points = len(sig1)
    cross_corr = signal.correlate(sig1 - np.mean(sig1), sig2 - np.mean(sig2), mode='full')
    cross_corr /= (np.std(sig1) * np.std(sig2) * n_points)  # Normalize
    lags = signal.correlation_lags(len(sig1),len(sig2), mode="full")
    return np.max(cross_corr), lags[np.argmax(cross_corr)]


def load_logger_data_new(folder_name, max_length, max_pop, n_trials = 100, RS = True):
    tcbk_list = []
    cell_list = []
    max_len = 0
    max_id = 0
    min_len = max_length
    min_id = 0
    for i in range(n_trials):
        fname = "trial_%d"%i
        try:
            with open(os.path.join(folder_name, str(fname)+"tcbk.pkl"), "rb") as f:
                info = pickle.load(f)
                tkbc = np.array(info["log"])
                # T_k_n0_applied = info["T_k_n0_applied"]
                # print(T_k_n0_applied)
        except:
            print("File not found: ", os.path.join(folder_name, str(fname)))
            continue

        # tcbk = tkbc[:,[0,3,2,1]].T
        if RS:
            tcbk = tkbc[:,[0,3,2,1,5,6]].T
        else:
            tcbk = tkbc[:,[0,3,2,1]].T
        tcbk_list.append(tcbk)
        cell_traj = expand_and_fill(tcbk[1,:], max_length)
        cell_list.append(cell_traj)

        if tcbk.shape[1] > max_len:
            max_len = tcbk.shape[1]
            max_id = i
        
        if tcbk[1,-1] > max_pop and tcbk.shape[1] < min_len:
            min_len = tcbk.shape[1]
            min_id = i
    
    if min_len < max_len:
        max_len = min_len
        max_id = min_id
    
    t = tcbk_list[max_id][0,:]
    b = tcbk_list[max_id][2,:]
    k_n0 = tcbk_list[max_id][3,:]
    # cell_ave = np.mean(np.array(cell_list), axis=0)
    cell_array = np.array(cell_list)
    
    return tcbk_list, t, b, k_n0, cell_array, (max_len, max_id)


def plot_single(
        loaded_logger,
        out_name,
        color_nut,
        more_antibiotic = False,
        more_nutrient = False,
):
    # folder_name = f"{sim_folder}/a{antibiotic_value:.2f}_n{nutrient_value:.2f}_value_check/{initialize_app}_{half_period}/"
    tcbk_list, t, b, k_n0, cell_array, (_, max_id) = loaded_logger
    # cell_ave = np.mean(cell_array, axis=0)
    
    figure, ax = plt.subplots(5,1)
    figure.subplots_adjust(hspace=.0)
    ax_num = 0
    # ax[0].set_title(f"antibiotic conc. max = {antibiotic_value}, nutrient conc. = {nutrient_value}")
    if more_antibiotic:
        for j in range(20):
            ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][2], color='gray', alpha=0.3, linewidth=1)
    ax[ax_num].plot(t,b, color=ANTIBIOTIC_COLOR)
    ax[ax_num].set_ylabel('Antibiotic')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].axvline(x=delta_t*embed_len, linestyle='--', color='k')

    ax_num += 1
    if more_nutrient:
        for j in range(20):
            ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][3], color='gray', alpha=0.3, linewidth=1)
    ax[ax_num].plot(t, k_n0, color=color_nut)
    ax[ax_num].set_ylabel('Nutrient')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].axvline(x=delta_t*embed_len, linestyle='--', color='k')

    ax_num += 1
    for j in range(20):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax[ax_num].plot(tcbk_list[j][0,:-1], tcbk_list[j][4,:-1], color='gray', alpha=0.3, linewidth=1)
    # ax[ax_num].plot(t, cell_ave[:len(t)], color=color_nut)
    ax[ax_num].plot(t, tcbk_list[max_id][4], color="black")
    ax[ax_num].set_ylabel(r'$\phi_R$')
    ax[ax_num].set_ylim(bottom=-0.1, top=0.45)
    # ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].set_yscale('log')

    ax_num += 1
    for j in range(20):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax[ax_num].plot(tcbk_list[j][0,:-1], tcbk_list[j][5,:-1], color='gray', alpha=0.3, linewidth=1)
    # ax[ax_num].plot(t, cell_ave[:len(t)], color=color_nut)
    ax[ax_num].plot(t, tcbk_list[max_id][5], color="black")
    ax[ax_num].set_ylabel(r'$\phi_S$')
    ax[ax_num].set_ylim(bottom=-0.1, top=0.35)
    # ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].set_yscale('log')

    ax_num += 1
    for j in range(20):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][1], color='gray', alpha=0.3, linewidth=1)
    # ax[ax_num].plot(t, cell_ave[:len(t)], color=POP_SIZE_COLOR)
    ax[ax_num].plot(t, tcbk_list[max_id][1], color=POP_SIZE_COLOR)
    ax[ax_num].set_ylabel(r'$P$')
    ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].set_yscale('log')

    # out_path = "figures/a%.2f_n%.2f_value_check/"%(antibiotic_value, nutrient_value)
    # extract path from out_name
    out_path = os.path.dirname(out_name)
    # out_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    figure.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(figure)


def plot_single_varenv(
        loaded_logger,
        out_name,
        line_color_list = None,
        n_trials = 100
):
    # folder_name = f"{sim_folder}/a{antibiotic_value:.2f}_n{nutrient_value:.2f}_value_check/{initialize_app}_{half_period}/"
    tcbk_list, t, b, k_n0, cell_array, (_, max_id) = loaded_logger
    # cell_ave = np.mean(cell_array, axis=0)
    
    np.random.seed(1)
    choices = np.random.choice(n_trials, 3)

    figure, ax = plt.subplots(5,1)
    figure.subplots_adjust(hspace=.0)
    ax_num = 0
    # ax[0].set_title(f"antibiotic conc. max = {antibiotic_value}, nutrient conc. = {nutrient_value}")
    for i_c, j in enumerate(choices):
        ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][2], color=line_color_list[i_c])
    ax[ax_num].set_ylabel('Antibiotic')
    ax[ax_num].get_xaxis().set_ticks([])

    ax_num += 1
    for i_c, j in enumerate(choices):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][3], color=line_color_list[i_c])
    ax[ax_num].set_ylabel('Nutrient')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].axvline(x=delta_t*embed_len, linestyle='--', color='k')

    ax_num += 1
    for i_c, j in enumerate(choices):
        ax[ax_num].plot(tcbk_list[j][0,:-1], tcbk_list[j][4,:-1], color=line_color_list[i_c])
    ax[ax_num].set_ylabel(r'$\phi_R$')
    ax[ax_num].set_ylim(bottom=-0.1, top=0.45)
    # ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].set_yscale('log')

    ax_num += 1
    for i_c, j in enumerate(choices):
        ax[ax_num].plot(tcbk_list[j][0,:-1], tcbk_list[j][5,:-1], color=line_color_list[i_c])
    ax[ax_num].set_ylabel(r'$\phi_S$')
    ax[ax_num].set_ylim(bottom=-0.1, top=0.35)
    # ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].get_xaxis().set_ticks([])
    # ax[ax_num].set_yscale('log')

    ax_num += 1
    for i_c, j in enumerate(choices):
        ax[ax_num].plot(tcbk_list[j][0], tcbk_list[j][1], color=line_color_list[i_c])
    ax[ax_num].set_ylabel(r'$P$')
    ax[ax_num].set_xlabel('Time (h)')
    ax[ax_num].set_yscale('log')

    # out_path = "figures/a%.2f_n%.2f_value_check/"%(antibiotic_value, nutrient_value)
    # extract path from out_name
    out_path = os.path.dirname(out_name)
    # out_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    figure.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(figure)


def plot_single_separate(
        loaded_logger,
        out_name,
        color_nut,
        more_antibiotic = False,
        more_nutrient = False,
        warm_up_embed = 61,
):
    # folder_name = f"{sim_folder}/a{antibiotic_value:.2f}_n{nutrient_value:.2f}_value_check/{initialize_app}_{half_period}/"
    tcbk_list, t, b, k_n0, _, (_, max_id) = loaded_logger

    out_path = os.path.dirname(out_name)
    out_name_type = str(out_name).split(".")[-1]
    out_name_base = str(out_name).split(f".{out_name_type}")[0]
    os.makedirs(out_path, exist_ok=True)

    figure, ax = plt.subplots(figsize=(8, 6))
    # ax[0].set_title(f"antibiotic conc. max = {antibiotic_value}, nutrient conc. = {nutrient_value}")
    if more_antibiotic:
        for j in range(20):
            ax.plot(tcbk_list[j][0], tcbk_list[j][2], color='gray', alpha=0.3, linewidth=1)
    ax.plot(t,b, color=ANTIBIOTIC_COLOR, linewidth=4)
    # ax.axvline(x=t[warm_up_embed], linestyle='--', color='k')
    ax.set_ylabel('Antibiotic', fontdict={'fontsize': 32})
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # ax.get_xaxis().set_ticks([])
    figure.savefig(f"{out_name_base}_antibiotic.{out_name_type}", dpi=300, bbox_inches='tight')
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 6))
    if more_nutrient:
        for j in range(20):
            ax.plot(tcbk_list[j][0], tcbk_list[j][3], color='gray', alpha=0.3, linewidth=1)
    ax.plot(t, k_n0, color=color_nut, linewidth=4)
    # ax.axvline(x=t[warm_up_embed], linestyle='--', color='k')
    ax.set_ylabel('Nutrient', fontdict={'fontsize': 32})
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # ax.get_xaxis().set_ticks([])
    figure.savefig(f"{out_name_base}_nutrient.{out_name_type}", dpi=300, bbox_inches='tight')
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 6))
    for j in range(20):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax.plot(tcbk_list[j][0,:-1], tcbk_list[j][4,:-1], color='gray', alpha=0.3, linewidth=1)
    ax.plot(t, tcbk_list[max_id][4], color="black", linewidth=4)
    # ax.axvline(x=t[warm_up_embed], linestyle='--', color='k')
    ax.set_ylabel(r'$\phi_R$', fontdict={'fontsize': 32})
    ax.set_ylim(bottom=-0.1, top=0.45)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # ax.get_xaxis().set_ticks([])
    figure.savefig(f"{out_name_base}_phiR.{out_name_type}", dpi=300, bbox_inches='tight')
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 6))
    for j in range(20):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax.plot(tcbk_list[j][0,:-1], tcbk_list[j][5,:-1], color='gray', alpha=0.3, linewidth=1)
    ax.plot(t, tcbk_list[max_id][5], color="black", linewidth=4)
    # ax.axvline(x=t[warm_up_embed], linestyle='--', color='k')
    ax.set_ylabel(r'$\phi_S$', fontdict={'fontsize': 32})
    ax.set_ylim(bottom=-0.1, top=0.35)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # ax.get_xaxis().set_ticks([])
    figure.savefig(f"{out_name_base}_phiS.{out_name_type}", dpi=300, bbox_inches='tight')
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 6))
    for j in range(20):  # Choose a subset (e.g., 20 out of 100) for shadow effect
        ax.plot(tcbk_list[j][0], tcbk_list[j][1], color='gray', alpha=0.3, linewidth=1)
    ax.plot(t, tcbk_list[max_id][1], color=POP_SIZE_COLOR, linewidth=4)
    ax.axvline(x=t[warm_up_embed], linestyle='--', color='k')
    ax.set_ylabel(r'$P$', fontdict={'fontsize': 16})
    ax.set_xlabel('Time (h)', fontdict={'fontsize': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_yscale('log')
    figure.savefig(f"{out_name_base}_pop.{out_name_type}", dpi=300, bbox_inches='tight')
    plt.close(figure)
