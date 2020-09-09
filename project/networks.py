'''
Network defs go in this file, so that they can be imported both from jupyter
(for training) and from the demo.
'''

import os
import re
import pickle
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .data import PFNN_XMEAN, PFNN_XSTD, PFNN_YMEAN, PFNN_YSTD, PFNN_YMEAN_ORIGINAL

print('CUDA Available:', torch.cuda.is_available())


def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_normal_(m.weight)
    # TODO: handle PFNN


# Base class for networks
class CharControlNet(nn.Module, ABC):
    # Path prefix for network checkpoints (redefine for each subclass)
    prefix = 'project/data/networks/'

    in_features = 342
    out_features = 311

    def __init__(self):
        super().__init__()

        self._best_loss = float('inf')

    @abstractmethod
    def forward(self, p, x):
        '''
        :param p: the phases for each example in batch
        :type p: torch Tensor of shape (batch_size,)
        :param x: the network inputs for each example in batch
        :type x: torch Tensor of shape (batch_size, in_features)
        :return: torch Tensor of shape (batch_size, out_features)
        '''
        raise NotImplementedError

    def inference(self, p, x):
        '''
        :param p: the phase
        :type p: float
        :param x: the network input
        :type x: numpy array of shape (in_features,)
        :return: numpy array of shape (out_features,)
        '''
        with torch.no_grad():
            p = torch.Tensor([p])
            x = torch.as_tensor(x).unsqueeze(0)
            return self.forward(p, x).reshape(-1).detach().numpy()

    def loss(self, prediction, label, reduction='mean'):
        return F.mse_loss(prediction, label, reduction=reduction)

    def save(self, model_version=None):
        directory = self._get_directory(model_version)
        checkpts = [int(f[:-3]) for f in os.listdir(directory)
                                if re.fullmatch('[0-9]+\.pt', f)]
        next_checkpt = max(checkpts)+1 if checkpts else 1
        fname = str(next_checkpt) + '.pt'
        torch.save(self.state_dict(), directory+fname)
        print(type(self).__name__+': Saved '+fname)

    def load(self, model_version=None):
        directory = self._get_directory(model_version)
        checkpts = [int(f[:-3]) for f in os.listdir(directory)
                                if re.fullmatch('[0-9]+\.pt', f)]
        if not checkpts:
            return 0

        last_checkpt = max(checkpts)
        fname = str(last_checkpt) + '.pt'
        print(type(self).__name__+': Loading '+fname)
        self.load_state_dict(torch.load(directory+fname))
        return last_checkpt

    def write_log(self, data, model_version=None):
        directory = self._get_directory(model_version)
        with open(directory+'log.pkl', 'wb') as f:
            pickle.dump(data, f)

    def read_log(self, model_version=None, default=([], [])):
        directory = self._get_directory(model_version)
        if 'log.pkl' in os.listdir(directory):
            with open(directory+'log.pkl', 'rb') as f:
                data = pickle.load(f)
            return data
        return default

    def save_if_best(self, loss, model_version=None):
        if loss < self._best_loss:
            self._best_loss = loss
            self.save(model_version)

    @classmethod
    def load_version(cls, model_version=None):
        net = cls()
        net.load(model_version)
        return net

    def _get_directory(self, model_version):
        directory = self.prefix + (model_version+'/' if model_version else '')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        return directory


class DummyNet(CharControlNet):
    ymean = torch.as_tensor(PFNN_YMEAN_ORIGINAL)

    def forward(self, p, x):
        y = torch.zeros(311)
        y[3] = 0.02
        y[26:32] = 1
        y[32:125] = self.ymean[32:125]
        y[218:311] = self.ymean[218:311]
        return y

    def load(self, model_version=None):
        pass


class LinearNet(CharControlNet):
    prefix = 'project/data/networks/linear/'

    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(1+self.in_features, self.out_features)

    def forward(self, p, x):
        # p is (batch_size,)
        # x is (batch_size, in_features)

        # Add the phase p as another feature in the network's input.
        x = torch.cat((p.unsqueeze(1), x), dim=1)  # (batch_size, 1+in_features)

        x = self.lin(x)                            # (batch_size, out_features)
        return x


class BasicReluNet(CharControlNet):
    prefix = 'project/data/networks/basicrelu/'

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(1+self.in_features, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, self.out_features)

    def forward(self, p, x):
        # p is (batch_size,)
        # x is (batch_size, in_features)

        # Add the phase p as another feature in the network's input.
        x = torch.cat((p.unsqueeze(1), x), dim=1)  # (batch_size, 1+in_features)

        x = F.relu(self.lin1(x))                   # (batch_size, 1024)
        x = F.relu(self.lin2(x))                   # (batch_size, 1024)
        x = self.lin3(x)                           # (batch_size, out_features)
        return x

    def inference(self, p, x):
        x -= PFNN_XMEAN
        x /= PFNN_XSTD
        y = super().inference(p, x)
        y *= PFNN_YSTD
        y += PFNN_YMEAN
        return y


class BasicEluNet(CharControlNet):
    prefix = 'project/data/networks/basicelu/'

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(1+self.in_features, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, self.out_features)

    def forward(self, p, x):
        # p is (batch_size,)
        # x is (batch_size, in_features)

        # Add the phase p as another feature in the network's input.
        x = torch.cat((p.unsqueeze(1), x), dim=1)  # (batch_size, 1+in_features)

        x = F.elu(self.lin1(x))                    # (batch_size, 1024)
        x = F.elu(self.lin2(x))                    # (batch_size, 1024)
        x = self.lin3(x)                           # (batch_size, out_features)
        return x

    def inference(self, p, x):
        x -= PFNN_XMEAN
        x /= PFNN_XSTD
        y = super().inference(p, x)
        y *= PFNN_YSTD
        y += PFNN_YMEAN
        return y


class DeepEluNet(CharControlNet):
    prefix = 'project/data/networks/deepelu/'

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(1+self.in_features, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 512)
        self.lin4 = nn.Linear(512, 512)
        self.lin5 = nn.Linear(512, self.out_features)

    def forward(self, p, x):
        # p is (batch_size,)
        # x is (batch_size, in_features)

        # Add the phase p as another feature in the network's input.
        x = torch.cat((p.unsqueeze(1), x), dim=1)  # (batch_size, 1+in_features)

        x = F.elu(self.lin1(x))                    # (batch_size, 1024)
        x = F.elu(self.lin2(x))                    # (batch_size, 1024)
        x = F.elu(self.lin3(x))                    # (batch_size, 512)
        x = F.elu(self.lin4(x))                    # (batch_size, 512)
        x = self.lin5(x)                           # (batch_size, out_features)
        return x

    def inference(self, p, x):
        x -= PFNN_XMEAN
        x /= PFNN_XSTD
        y = super().inference(p, x)
        y *= PFNN_YSTD
        y += PFNN_YMEAN
        return y


class PhaseFunctionedNet(CharControlNet):
    prefix = 'project/data/networks/pfnn/'

    def __init__(self):
        super().__init__()

        self.W0 = nn.Parameter(torch.zeros((4, self.in_features, 512)))
        self.b0 = nn.Parameter(torch.zeros((4, 512)))

        self.W1 = nn.Parameter(torch.zeros((4, 512, 512)))
        self.b1 = nn.Parameter(torch.zeros((4, 512)))

        self.W2 = nn.Parameter(torch.zeros((4, 512, self.out_features)))
        self.b2 = nn.Parameter(torch.zeros((4, self.out_features)))

    def forward(self, p, x):
        print('forward')
        # p is (batch_size,)
        # x is (batch_size, in_features)

        p = p*4

        pa, pindex_1 = p % 1.0, p.to(torch.int64) % 4
        pindex_0 = (pindex_1-1) % 4
        pindex_2 = (pindex_1+1) % 4
        pindex_3 = (pindex_1+2) % 4

        def cubic(y0, y1, y2, y3, mu, mu2, mu3):
            ret = (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu3
            ret += (y0-2.5*y1+2.0*y2-0.5*y3)*mu2
            ret += (-0.5*y0+0.5*y2)*mu
            ret += y1
            return ret

        wa = pa.unsqueeze(1).unsqueeze(2)
        wa2 = wa*wa
        wa3 = wa2*wa

        W0 = cubic(self.W0.data[pindex_0], self.W0.data[pindex_1], self.W0.data[pindex_2],
                   self.W0.data[pindex_3], wa, wa2, wa3)
        W1 = cubic(self.W1.data[pindex_0], self.W1.data[pindex_1], self.W1.data[pindex_2],
                   self.W1.data[pindex_3], wa, wa2, wa3)
        W2 = cubic(self.W2.data[pindex_0], self.W2.data[pindex_1], self.W2.data[pindex_2],
                   self.W2.data[pindex_3], wa, wa2, wa3)

        ba = pa.unsqueeze(1)
        ba2 = ba*ba
        ba3 = ba2*ba

        b0 = cubic(self.b0.data[pindex_0], self.b0.data[pindex_1], self.b0.data[pindex_2],
                   self.b0.data[pindex_3], ba, ba2, ba3)
        b1 = cubic(self.b1.data[pindex_0], self.b1.data[pindex_1], self.b1.data[pindex_2],
                   self.b1.data[pindex_3], ba, ba2, ba3)
        b2 = cubic(self.b2.data[pindex_0], self.b2.data[pindex_1], self.b2.data[pindex_2],
                   self.b2.data[pindex_3], ba, ba2, ba3)

        # Pass x through network
        x = x.unsqueeze(1)                              # (batch_size, 1, in_features)
        x = F.elu(torch.bmm(x, W0) + b0.unsqueeze(1))   # (batch_size, 1, 512)
        x = F.elu(torch.bmm(x, W1) + b1.unsqueeze(1))   # (batch_size, 1, 512)
        x =       torch.bmm(x, W2) + b2.unsqueeze(1)    # (batch_size, 1, out_features)
        x = x.squeeze(1)                                # (batch_size, out_features)
        return x

    def inference(self, p, x):
        x -= PFNN_XMEAN
        x /= PFNN_XSTD
        y = super().inference(p, x)
        y *= PFNN_YSTD
        y += PFNN_YMEAN
        return y


class FastPFNN(PhaseFunctionedNet):
    prefix = 'project/data/networks/fastpfnn/'

    def __init__(self, dropout=0, input_dropout=False):
        super().__init__()

        self.lin1 = nn.Linear(self.in_features, 2048)
        self.lin2 = nn.Linear(512, 2048)
        self.lin3 = nn.Linear(512, 4*self.out_features)

        self.dropout = nn.Dropout(dropout)

        self.input_dropout = input_dropout

    def forward(self, p, x):
        # p is (batch_size,)
        # x is (batch_size, in_features)

        p = p*4
        pshift = (p.to(torch.int64) - 1) % 4

        pa = torch.empty((p.shape[0], 4), device=p.device)
        pa[:,0] = 1
        pa[:,1] = p % 1.0
        pa[:,2] = pa[:,1]*pa[:,1]
        pa[:,3] = pa[:,2]*pa[:,1]

        interp_weights = pa @ torch.tensor([
            [ 0.0,  1.0,  0.0,  0.0],
            [-0.5,  0.0,  0.5,  0.0],
            [ 1.0, -2.5,  2.0, -0.5],
            [-0.5,  1.5, -1.5,  0.5]
        ], device=p.device)
        interp_weights[pshift == 1] = interp_weights[pshift == 1].roll(1, 1)
        interp_weights[pshift == 2] = interp_weights[pshift == 2].roll(2, 1)
        interp_weights[pshift == 3] = interp_weights[pshift == 3].roll(3, 1)

        def catmull_rom_interp(x, num_features):
            x = x.reshape(x.shape[0], num_features, 4)
            return torch.bmm(x, interp_weights.unsqueeze(2)).squeeze(2)

        # Pass x through network
        if self.input_dropout: x = self.dropout(x)
        x = F.elu(self.dropout(catmull_rom_interp(self.lin1(x), 512)))  # (batch_size, 512)
        x = F.elu(self.dropout(catmull_rom_interp(self.lin2(x), 512)))  # (batch_size, 512)
        x = catmull_rom_interp(self.lin3(x), self.out_features)         # (batch_size, out_features)
        return x

    def export(self, path):

        W0n = np.moveaxis(self.lin1.weight.data.detach().numpy().reshape(self.in_features, 512, 4), 2, 0)
        W1n = np.moveaxis(self.lin2.weight.data.detach().numpy().reshape(512, 512, 4), 2, 0)
        W2n = np.moveaxis(self.lin3.weight.data.detach().numpy().reshape(512, self.out_features, 4), 2, 0)

        b0n = np.moveaxis(self.lin1.bias.data.detach().numpy().reshape(512, 4), 1, 0)
        b1n = np.moveaxis(self.lin2.bias.data.detach().numpy().reshape(512, 4), 1, 0)
        b2n = np.moveaxis(self.lin3.bias.data.detach().numpy().reshape(self.out_features, 4), 1, 0)

        for i in range(50):

            pscale = 4*(float(i)/50)
            pamount = pscale % 1.0

            pindex_1 = int(pscale) % 4
            pindex_0 = (pindex_1-1) % 4
            pindex_2 = (pindex_1+1) % 4
            pindex_3 = (pindex_1+2) % 4

            def cubic(y0, y1, y2, y3, mu):
                return (
                    (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu +
                    (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu +
                    (-0.5*y0+0.5*y2)*mu +
                    (y1))

            W0 = cubic(W0n[pindex_0], W0n[pindex_1], W0n[pindex_2], W0n[pindex_3], pamount)
            W1 = cubic(W1n[pindex_0], W1n[pindex_1], W1n[pindex_2], W1n[pindex_3], pamount)
            W2 = cubic(W2n[pindex_0], W2n[pindex_1], W2n[pindex_2], W2n[pindex_3], pamount)

            b0 = cubic(b0n[pindex_0], b0n[pindex_1], b0n[pindex_2], b0n[pindex_3], pamount)
            b1 = cubic(b1n[pindex_0], b1n[pindex_1], b1n[pindex_2], b1n[pindex_3], pamount)
            b2 = cubic(b2n[pindex_0], b2n[pindex_1], b2n[pindex_2], b2n[pindex_3], pamount)

            W0.astype(np.float32).tofile(path + 'W0_%03i.bin' % i)
            W1.astype(np.float32).tofile(path + 'W1_%03i.bin' % i)
            W2.astype(np.float32).tofile(path + 'W2_%03i.bin' % i)

            b0.astype(np.float32).tofile(path + 'b0_%03i.bin' % i)
            b1.astype(np.float32).tofile(path + 'b1_%03i.bin' % i)
            b2.astype(np.float32).tofile(path + 'b2_%03i.bin' % i)


class StockPFNN(CharControlNet):
    def forward(self, p, x):
        i = int(p * 50)
        h0 = F.elu(x @ self.W0[i] + self.b0[i])
        h1 = F.elu(h0 @ self.W1[i] + self.b1[i])
        y = h1 @ self.W2[i] + self.b2[i]
        return y

    def inference(self, p, x):
        x -= self.Xmean
        x /= self.Xstd
        y = super().inference(p, x)
        y *= self.Ystd
        y += self.Ymean
        return y

    def save(self, model_version=None):
        raise NotImplementedError

    def load(self, model_version=None):
        self.Xmean = np.fromfile('demo/network/pfnn/Xmean.bin', dtype=np.float32)
        self.Xstd = np.fromfile('demo/network/pfnn/Xstd.bin', dtype=np.float32)
        self.Ymean = np.fromfile('demo/network/pfnn/Ymean.bin', dtype=np.float32)
        self.Ystd = np.fromfile('demo/network/pfnn/Ystd.bin', dtype=np.float32)

        W0 = np.zeros((50, self.in_features, 512), dtype=np.float32)
        W1 = np.zeros((50, 512, 512), dtype=np.float32)
        W2 = np.zeros((50, 512, self.out_features), dtype=np.float32)
        b0 = np.zeros((50, 512), dtype=np.float32)
        b1 = np.zeros((50, 512), dtype=np.float32)
        b2 = np.zeros((50, self.out_features), dtype=np.float32)

        for i in range(50):
            W0[i] = np.fromfile(f'demo/network/pfnn/W0_{i:03d}.bin', dtype=np.float32).reshape(512, self.in_features).T
            W1[i] = np.fromfile(f'demo/network/pfnn/W1_{i:03d}.bin', dtype=np.float32).reshape(512, 512).T
            W2[i] = np.fromfile(f'demo/network/pfnn/W2_{i:03d}.bin', dtype=np.float32).reshape(self.out_features, 512).T
            b0[i] = np.fromfile(f'demo/network/pfnn/b0_{i:03d}.bin', dtype=np.float32)
            b1[i] = np.fromfile(f'demo/network/pfnn/b1_{i:03d}.bin', dtype=np.float32)
            b2[i] = np.fromfile(f'demo/network/pfnn/b2_{i:03d}.bin', dtype=np.float32)

        self.W0 = torch.as_tensor(W0)
        self.W1 = torch.as_tensor(W1)
        self.W2 = torch.as_tensor(W2)
        self.b0 = torch.as_tensor(b0)
        self.b1 = torch.as_tensor(b1)
        self.b2 = torch.as_tensor(b2)
