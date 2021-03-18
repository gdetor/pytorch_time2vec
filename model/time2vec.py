# time2vec class
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

from torch import nn, cat, device
from torch import bmm, sin, rand


class Time2Vec(nn.Linear):
    """
        time2vec_layer class implements a Time2Vec layer based on the work:
        [1] "Time2Vec: Learning a Vector Representation of Time",
             Kazemi et al., 2019.
    """
    def __init__(self, in_features, out_features, bias=True, func=sin,
                 dev="cuda:0"):
        """
            Constructor of time2vec_layer.

            Args:
                in_features (int):  Number of input features
                out_features (int): Number of output features
                bias (bool):        True to add bias - False to not add bias
                                    term.
                func (object):      Torch periodic function (sin, cos, etc)
                dev (str):          Computational Device (GPU or CPU)

            Returns:

        """
        super(Time2Vec, self).__init__(in_features, out_features, bias)
        self.out_features = out_features
        self.dev = device(dev)

        # Define and register the essential Time2Vec parameters
        # w0 and b0 correspond to omega_0 and phi_0 in [1]
        # self.W0 = nn.Parameter(rand(1, 1).cuda())
        self.W0 = nn.Parameter(rand(1, 1).to(self.dev))
        self.register_parameter("W0", self.W0)
        self.b0 = nn.Parameter(rand(in_features, 1).to(self.dev))
        # self.b0 = nn.Parameter(rand(in_features, 1).cuda())
        self.register_parameter("b0", self.b0)

        # W and b correspond to omega_i and phi_i for i != 0
        # omega_i and phi_i represent the frequency and phase-shift in case
        # func=sin.
        self.W = nn.Parameter(rand(out_features,
                                   out_features).to(self.dev))
        # self.W = nn.Parameter(rand(out_features,
        #                            out_features).cuda())
        self.register_parameter("W", self.W)
        self.b = nn.Parameter(rand(in_features,
                                   out_features).to(self.dev))
        # self.b = nn.Parameter(rand(in_features,
        #                            out_features).cuda())
        self.register_parameter("b", self.b)

        # Nonlinear period function
        self.f = func

        # Disable gradients for intrinsic weights and biases (see nn.Linear)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, tau):
        """
            Time2Vec layer forward method. The input signal (tensor) contains
            a sequence with dimension -> features_dim.

            Args:
                tau (tensor):       Input signal (torch tensor)
                                    (batch_size, sequence_length, features_dim)

            Returns:
                A tensor that contains the concatenation of a predictive and a
                progressive signal, respectively.

            Notes:
                For instance, if the time2vec layer is being used in timeseries
                prediction the sequence length is the length of the timeseries
                chunk and the features dimension is 1 if the prediction horizon
                is one.
        """
        batch_size = tau.shape[0]
        W = self.W.repeat(batch_size, 1, 1)
        b = self.b.repeat(batch_size, 1, 1)
        # Progressive signal
        res_progressive = tau * self.W0 + self.b0
        x = tau.repeat_interleave(self.out_features, dim=-1)
        # Predictive signal
        res_predictive = self.f(bmm(x, W) + b)
        out = cat([res_predictive, res_progressive], 2)
        return out
