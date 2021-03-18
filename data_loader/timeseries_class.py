# Time series Pytorch DataLoader Class
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
from numpy import load, float32, random, expand_dims
from numpy import nan_to_num, log, abs, sign, isnan, count_nonzero
from torch import from_numpy, is_tensor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


from scipy.stats import boxcox


def mu_law(x, mu=255):
    """ Compute the mu law (companding algorithm). It reduces the dynamic
    range of the input signal x. This can be used in cases where the original
    signal x has, for instance, 16-bit integer values and we need to pass it
    through a softmax layer (as in Wavenet) to get all possible probabilities
    [65536 (2**16)]. Passing the signal through mu_law it will encode
    (compress) the values to 256 (if mu=255).
    """
    mulaw = sign(x) * log(1 + mu * abs(x)) / log(1 + mu)
    return mulaw


class timeseries(Dataset):
    """ Timeseries Class. This class preprocesses timeseries data and provides
    a Pytorch DataLoader for training and testing models.
    """
    def __init__(self, fname, win_size=10, horizon=1, dim_size=1, scale=False,
                 ab=(0, 1), standarize=False, power_transform=False,
                 train=True, noise=False, var=1.0, mulaw=False):
        """
        Constructor method of timeseries class.

        Args:
            fname (string):     Raw data filename (must be numpy file-npy)
            win_size (int):     Window size (chunk of timeseries)
            horizon (int):      Prediction horizon (default=1)
            dim_size (int):     (Features dimension or number of variables in
                                case of multivariable timeseries)
            scale (bool):       Scale the raw data
            ab (floats tuple):  Interval to scale the raw data
            standarize (bool):  Standarize the raw data
            power_transform (bool): Apply a power transform to the raw data
            train (bool):       True to split the raw data into two sets
                                (80% of the raw data for training)
            noise (bool):       Add white noise to the data
            var (float):        Variance of the additive white noise (noise
                                argument should be True)
            mulaw (bool):       Apply a mu-law companding algorithm

        Returns:

        """
        self.horizon = horizon
        # Load the data
        data = load(fname).astype(float32)
        if data.ndim != 1:
            data = data[:, :dim_size]

        # Check for NaNs
        if count_nonzero(isnan(data)):
            print("WARNING: NaN detected in the raw data!")

        # Remove NaNs
        data = nan_to_num(data, nan=0.0).astype(float32)

        perc = int(data.shape[0] * 0.7)
        if train is True:
            data = data[:perc]
        else:
            data = data[perc:]
        self.shape = data.shape             # Keep data shape
        self.win_len = win_size             # Prediction horizon

        # Add white noise to the data
        if noise is True:
            data += random.normal(0, var, data.shape)

        # Convert data Numpy array to Torch Tensor
        self.data = from_numpy(data)

        # Ensure the data are positive when Box-Cox transform is enabled
        if scale is False and power_transform is True:
            scale = True

        # Scale the data [0, 1]
        if scale:
            scaler = MinMaxScaler(feature_range=(ab[0], ab[1]), copy=True)
            if len(self.data.shape) == 1:
                self.data = scaler.fit_transform(self.data.reshape(-1, 1))
                self.data = self.data[:, 0]
            else:
                self.data = scaler.fit_transform(self.data)
            self.data = self.data.astype(float32)

        # Standarize the data (x - mu) /  sigma
        if standarize is True:
            standarizer = StandardScaler()
            if len(self.data.shape) == 1:
                self.data = standarizer.fit_transform(self.data.reshape(-1, 1))
                self.data = self.data[:, 0]
            else:
                self.data = standarizer.fit_transform(self.data)
            self.data = self.data.astype(float32)

        # Apply a Box-Cox (power) transform
        if power_transform is True:
            self.data, lamda = boxcox(self.data.flatten())
            self.data = self.data.reshape(self.shape)

        if mulaw is True:
            self.data = mu_law(self.data)

        # Final data tensor length
        self.size = len(self.data) - (win_size + 1)

    def __len__(self):
        """ Return the length of data. """
        return len(self.data)

    def __getitem__(self, idx):
        """ Get an item from data. Slide the window based on the horizon.
        """
        if is_tensor(idx):
            idx = idx.tolist()
        idx %= self.size
        x = self.data[idx:idx+self.win_len]
        if self.horizon == 1:
            y = self.data[idx+self.win_len]
        else:
            y = self.data[idx+1:idx+self.win_len+1]
        if self.data.ndim == 1:
            x = expand_dims(x, axis=1)
            y = expand_dims(y, axis=0)
        return x, y, idx
