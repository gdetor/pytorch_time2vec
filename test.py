import numpy as np
import torch
from model.time2vec import Time2Vec

np.random.seed(13)
torch.manual_seed(13)


if __name__ == '__main__':
    dev = torch.device("cpu")
    x = torch.from_numpy(np.ones((2, 3, 1), 'f'))
    tv = Time2Vec(1, 3, dev=dev)
    res = tv(x)
    print(res.shape)
    print(res)
