# LSTM and T2V-LSTM classes
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
from torch import nn, randn
from .time2vec import Time2Vec


def init_weights(layer):
    """
    Initialize weights using Xavier's method (uniform).

    Args:
        layer (torch object):   Torch Layer

    Returns:

    """
    for w in layer._all_weights:
        for p in w:
            if 'weight' in p:
                nn.init.xavier_uniform_(layer.__getattr__(p).data)


class T2VLSTM(nn.Module):
    """
    LSTM class equipped with Time2Vec layer. The input is firstly passed to
    the Time2Vec layer and then is forwarded to the LSTM.
    """
    def __init__(self, t2v_size, input_size, hidden_size, num_layers,
                 seq_len=1, dev='cpu'):
        """
        Constructor of T2VLSTM class

        Args:
            t2v_size (int):     Number of units in Time2Vec layer
            input_size (int):   Input size of LSTM
            hidden_size (int):  Number of units per hidden layer of LSTM
            num_layers (int):   Number of LSTM layers
            seq_len (int):      Sequence length
            dev (torch device): CPU or GPU device

        Returns:

        """
        super(T2VLSTM, self).__init__()
        self.n_layers = num_layers
        self.hidden_dim = hidden_size
        self.dev = dev

        # Time2Vec Layer
        self.t2v = Time2Vec(seq_len, t2v_size, dev=self.dev)
        # LSTM
        self.lstm = nn.LSTM(t2v_size+1, hidden_size, 1, batch_first=True)
        # FC layer
        self.fc = nn.Linear(hidden_size * seq_len, 1)

        # Initialize all layers
        nn.init.xavier_uniform_(self.t2v.W)
        nn.init.xavier_uniform_(self.t2v.W0)
        nn.init.uniform_(self.t2v.b0, -0.01, 0.01)
        nn.init.uniform_(self.t2v.b, -0.01, 0.01)
        nn.init.xavier_uniform_(self.fc.weight)

        # LSTM Initialization - Weights and Bias
        for layer in self.lstm._all_weights:
            for p in layer:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.lstm.__getattr__(p).data)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = self.lstm.__getattr__(name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        # Flag is used for initializing the states of LSTM (h0, c0)
        self.flag = 0

    def forward(self, x):
        """
        Forward method of T2VLSTM class.

        Args:
            x (tensor): Input tensor (batch_size, sequence_lengh, features_dim)

        Returns:
            Tensor of size (batch_size, sequence_lengh, features_dim)
        """
        batch_size = x.shape[0]
        # Initialize LSTM states
        if self.flag == 0:
            self.h0 = nn.Parameter(randn(self.n_layers*1, batch_size,
                                         self.hidden_dim),
                                   requires_grad=True).to(self.dev)
            self.c0 = nn.Parameter(randn(self.n_layers*1, batch_size,
                                         self.hidden_dim),
                                   requires_grad=True).to(self.dev)
            self.flag = 1
        # Pass the input signal through Time2Vec layer
        out = self.t2v(x)
        out, (self.h, self.c) = self.lstm(out, (self.h0, self.c0))
        m, n = out.shape[1], out.shape[2]
        out = out.reshape(-1, m * n)
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    """
    Standard LSTM class
    """
    def __init__(self, input_size, hidden_size, num_layers, seq_len=1,
                 dev='cpu'):
        """
        Constructor of LSTM class

        Args:
            input_size (int):   Input size of LSTM
            hidden_size (int):  Number of units per hidden layer of LSTM
            num_layers (int):   Number of LSTM layers
            seq_len (int):      Sequence length
            dev (torch device): CPU or GPU device

        Returns:

        """
        super(LSTM, self).__init__()
        self.n_layers = num_layers
        self.hidden_dim = hidden_size
        self.dev = dev

        # Define LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        # Define a FC layer
        self.fc = nn.Linear(hidden_size * seq_len, 1)

        # Initialize FC layer weights
        nn.init.xavier_uniform_(self.fc.weight)

        # Initialize LSTM's weights and biases
        for layer in self.lstm._all_weights:
            for p in layer:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.lstm.__getattr__(p).data)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = self.lstm.__getattr__(name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        # Flag is used for initializing the states of LSTM (h0, c0)
        self.flag = 0

    def forward(self, x):
        """
        Forward method of LSTM class.

        Args:
            x (tensor): Input tensor (batch_size, sequence_lengh, features_dim)

        Returns:
            Tensor of size (batch_size, sequence_lengh, features_dim)
        """
        batch_size = x.shape[0]
        # Initialize LSTM states
        if self.flag == 0:
            self.h0 = nn.Parameter(randn(self.n_layers*1, batch_size,
                                         self.hidden_dim),
                                   requires_grad=True).to(self.dev)
            self.c0 = nn.Parameter(randn(self.n_layers*1, batch_size,
                                         self.hidden_dim),
                                   requires_grad=True).to(self.dev)
            self.flag = 1
        out, (self.h, self.c) = self.lstm(x, (self.h0, self.c0))
        m, n = out.shape[1], out.shape[2]
        out = out.reshape(-1, m*n)
        out = self.fc(out)
        return out
