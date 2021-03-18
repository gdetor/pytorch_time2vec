from torch import nn
from model.time2vec import Time2Vec


class MLP(nn.Module):
    """
    Multi-Layer Perceptron class for time series forecasting.
    """
    def __init__(self, seq_len=1):
        """
        Constructor of MLP class.

        Args:
            seq_len (int):  Sequence length
            tv_dim (int):   Time2Vec dimension

        Returns:

        """
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(seq_len, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward method of MLP class.

        Args:
            x (torch tensor):   The input sequence (sequence length x number
                                                    of features)

        Returns:
            A pytorch tensor that contains the prediction.
        """
        x = x.view(-1, x.shape[1] * x.shape[2])
        out = self.relu(self.fc1(x))
        out = self.bn1(out)
        out = self.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.tanh(self.fc3(out))
        return out


class T2VMLP(nn.Module):
    """
    Multi-Layer Perceptron class for time series forecasting.
    """
    def __init__(self, seq_len=1, tv_dim=100, dev="cuda:0"):
        """
        Constructor of MLP class.

        Args:
            seq_len (int):  Sequence length
            tv_dim (int):   Time2Vec dimension

        Returns:

        """
        super(T2VMLP, self).__init__()

        self.t2v = Time2Vec(seq_len, tv_dim, dev=dev)
        self.fc1 = nn.Linear((tv_dim+1)*seq_len, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward method of MLP class.

        Args:
            x (torch tensor):   The input sequence
                                (batch_size, seq_length, number of features)

        Returns:
            A pytorch tensor that contains the prediction.
        """
        out = self.relu(self.t2v(x))
        out = out.reshape(-1, out.shape[1] * out.shape[2])
        out = self.relu(self.fc1(out))
        out = self.bn1(out)
        out = self.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.tanh(self.fc3(out))
        return out
