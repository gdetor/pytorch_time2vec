# Example of how to use Time2Vec Pytorch implementation.
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
import sys
import argparse
from numpy import array, save

import torch
from torch import nn, device, no_grad, manual_seed, backends
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_loader.timeseries_class import timeseries
from model.lstm import T2VLSTM, LSTM
from model.mlp import MLP, T2VMLP

import matplotlib.pylab as plt

backends.cudnn.deterministic = True
backends.cudnn.benchmark = False


manual_seed(135)


def run_experiment(model, data_path, sequence_length, epochs, batch_size,
                   cdevice):
    dev = device(cdevice)

    ts = timeseries(data_path, win_size=sequence_length,
                    scale=False, standarize=False, train=True)
    dataloader = DataLoader(ts, batch_size=batch_size, shuffle=False,
                            drop_last=True)

    if model == 'tvlstm':
        print("Time2Vec - LSTM")
        net = T2VLSTM(128, 1, 32, 1, sequence_length, dev=dev).to(dev)
    elif model == 'lstm':
        print("LSTM")
        net = LSTM(1, 32, 1, sequence_length, dev=dev).to(dev)
    elif model == 'mlp':
        print("MLP")
        net = MLP(seq_len=sequence_length).to(dev)
    elif model == 'tvmlp':
        print("T2VMLP")
        net = T2VMLP(seq_len=sequence_length, tv_dim=100, dev=dev).to(dev)
        print(net)
    else:
        print("Model is not specified!")
        sys.exit(-1)

    optimizer = Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Start training the model ... ")
    length_data = len(dataloader)
    loss_track = []
    for e in range(epochs):
        total_loss = 0
        for x, y, _ in dataloader:
            x = x.to(dev)
            y = y.to(dev)

            optimizer.zero_grad()

            y_hat = net(x)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_track.append(total_loss / length_data)
        print("[Epoch: %d Total Loss: %f]" % (e, loss_track[e]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_track, 'k', lw=2)
    ax.set_xlabel("Epochs", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    plt.show()

    print("Testing the trained model ...")
    ts = timeseries(data_path, win_size=sequence_length,
                    scale=False, standarize=False, train=False)
    dataloader = DataLoader(ts, batch_size=batch_size, shuffle=False,
                            drop_last=True)

    net.eval()
    with no_grad():
        error = 0
        y_pred = []
        for x, y, _ in dataloader:
            x = x.to(dev)
            y = y.to(dev)

            y_hat = net(x)
            tmp_error = nn.functional.l1_loss(y_hat, y)
            error += tmp_error.item()
            y_pred.append(y_hat.detach().cpu().numpy())
        error /= len(dataloader)
        print(error)
        if model == 'tvlstm':
            save("results/tvlstm_prediction", array(y_pred))
        elif model == 'lstm':
            save("results/lstm_prediction", array(y_pred))
        elif model == 'mlp':
            save("results/mlp_prediction", array(y_pred))
        elif model == 'tvmlp':
            save("results/tvmlp_prediction", array(y_pred))
        else:
            print("Nothing stored!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time2Vec Pytorch Impl")
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--sequence-len',
                        type=int,
                        default=23,
                        help='historical data points')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='batch size')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help="computational device (GPU or CPU)")
    parser.add_argument('--model',
                        type=str,
                        default='tvlstm',
                        help='model type (lstm, tvlstm, mlp, tvmlp)')
    parser.add_argument('--data-path',
                        type=str,
                        default='./data/livelo.npy',
                        help='the path where the input data can be found')
    args = parser.parse_args()

    if args.device == "cuda:0" and torch.cuda.is_available() is not True:
        print("There is no CUDA device available!")
        print("Fallback to CPU ... ")
        args.device = "cpu"

    run_experiment(args.model,
                   args.data_path,
                   args.sequence_len,
                   args.epochs,
                   args.batch_size,
                   args.device)
