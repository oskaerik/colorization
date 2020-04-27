import re
import io
import sys
import torch
from colorize import network, util

with open('colorize/network.py', 'r') as f:
    layer_parameters = f.read().splitlines()
layer_parameters = [line.strip() for line in layer_parameters]
layer_parameters = [line.split('self.')[1] for line in layer_parameters if line.startswith('self.conv')]
layer_parameters = ['Input'] + layer_parameters

net = network.Network()

# Save summary output to a string buffer
sys.stdout = buffer = io.StringIO()
net(torch.zeros((1, 1, *util.input_size)), summary=True)
sys.stdout = sys.__stdout__

layer_sizes = buffer.getvalue().splitlines()
layer_sizes = [line.split('torch.Size')[1] for line in layer_sizes]
layer_sizes = [''.join(c for c in line if c not in '[]()') for line in layer_sizes]
layer_sizes = [line.split(', ')[1:] for line in layer_sizes]

with open('architecture.csv', 'w') as f:
    print('T,X,C,K,S,P,D')
    print('T,X,C,K,S,P,D', file=f)
    size_generator = iter(layer_sizes)
    size = next(size_generator)
    C = 1
    for parameter in layer_parameters:
        if 'Conv2d' in parameter or 'ConvTranspose2d' in parameter:
            size = next(size_generator)
            pattern = re.compile(r'out_channels=(\d+), kernel_size=(\d+), stride=(\d+), padding=(\d+), dilation=(\d+)')
            result = pattern.search(parameter)
            T = 'Convolution' if 'Conv2d' in parameter else 'Deconvolution'
            C, K, S, P, D = result.group(1, 2, 3, 4, 5)
        elif 'Input' in parameter or 'BatchNorm2d' in parameter:
            T = 'Input' if 'Input' in parameter else 'BatchNorm'
            K = S = P = D = '-'
        X = f'${size[1]} \\times {size[2]}$'
        print(T, X, C, K, S, P, D, sep=',')
        print(T, X, C, K, S, P, D, sep=',', file=f)
