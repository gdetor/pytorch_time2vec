# Results analysis script
# This script is based on the implementation provided by Marco Cerliani
# https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
# Copyright (C) 2020  Georgios Is. Detorakis (gdetor@protonmail.com)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import matplotlib.pylab as plt


def bootstraping(x):
    """
    Bootstrapping (random sampling with replacement) to estimate the sampling
    distribution of x (input data).

    Args:
        x (ndarray):    Input data

    Returns:
        Bootstrapped sample distribution as Nump array.
    """
    sample = []
    for _ in range(1000):
        sample_mean = np.random.choice(x, 100).mean()
        sample.append(sample_mean)
    return np.array(sample)


if __name__ == '__main__':
    y_lstm = np.load("./results/lstm_prediction.npy")[:, :, 0].flatten()
    y_tvlstm = np.load("./results/tvlstm_prediction.npy")[:, :, 0].flatten()

    sample_lstm = []
    for _ in range(1000):
        sample_mean = np.random.choice(y_lstm, 100).mean()
        sample_lstm.append(sample_mean)

    sample_tvlstm = []
    for _ in range(1000):
        sample_mean = np.random.choice(y_tvlstm, 100).mean()
        sample_tvlstm.append(sample_mean)

    # Estimate the quantiles of the Bootstraped distributions in order to
    # check for distributions overlapping.
    print("LSTM Quantile: %f" % (np.quantile(sample_lstm, 0.9)))
    print("T2VLSTM Quantile: %f" % (np.quantile(sample_tvlstm, 0.1)))

    # Plot prediction distributions
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.hist(y_tvlstm, bins=40, alpha=0.5, color='k', label="T2VLSTM")
    ax.hist(y_lstm, bins=40, alpha=0.5, color='m', label="LSTM")
    ax.legend()
    ax.grid()
    plt.show()

    # Plot bootstraped prediction distributions
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.hist(sample_tvlstm, bins=40, alpha=0.5, color='k', label="T2VLSTM")
    ax.hist(sample_lstm, bins=40, alpha=0.5, color='m', label="LSTM")
    ax.legend()
    ax.grid()
    plt.show()
