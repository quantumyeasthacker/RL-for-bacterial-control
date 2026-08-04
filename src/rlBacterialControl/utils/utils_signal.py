"""Signal-analysis helpers shared by the agents (MLP_full, RNN_full)."""

import numpy as np
from scipy import signal


def cross_correlation(sig1, sig2, max_cross_corr, lag):
    n_points = len(sig1)
    cross_corr = signal.correlate(sig1 - np.mean(sig1), sig2 - np.mean(sig2), mode='full')
    cross_corr /= (np.std(sig1) * np.std(sig2) * n_points)  # Normalize
    max_cross_corr.append(np.max(cross_corr))
    lags = signal.correlation_lags(len(sig1), len(sig2), mode="full")
    lag.append(lags[np.argmax(cross_corr)])
