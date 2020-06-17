# -*- coding: utf-8 -*-

from sgn.gan import Generator
from argparse import ArgumentParser

import torch
import os
import json
import numpy as np


parser = ArgumentParser('Structure generation.')
parser.add_argument('--checkpoint', default='results/sgn/checkpoints/20000.pth.tar')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--sample_num', type=int, default=500)
parser.add_argument('--sample_dir', default='results/sgn/samples')


def matrix_to_structure(prob_matrix, name):
    prob_matrix_edge = (prob_matrix > 0.5).astype(np.int)
    edges = {}
    for c in range(2):
        for i in range(1, prob_matrix.shape[1]):
            for j in range(i):
                if prob_matrix_edge[c, i, j] == 1:
                    key = i
                    if key in edges.keys():
                        edges[key].append((j, c, prob_matrix[c, i, j]))
                    else:
                        edges[key] = [(j, c, prob_matrix[c, i, j])]

    edge_list = []
    for key in edges:
        probs = [x[2] for x in edges[key]]
        idx = np.argmax(probs)
        edge_list.append((edges[key][idx][0], key, edges[key][idx][1]))

    with open(name + '.json', 'w') as f:
        json.dump(edge_list, f)


def save_batch(samples, prefix):
    samples = samples.squeeze()
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for i in range(samples.shape[0]):
        matrix_to_structure(samples[i], os.path.join(prefix, '{}'.format(i)))


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    pretrained_checkpoint = torch.load(args.checkpoint)

    netG = Generator(dim=32)
    netG.load_state_dict(pretrained_checkpoint['G'])

    if args.use_cuda:
        netG = netG.cuda()

    generate_num = 0

    samples_list = []
    while args.sample_num > generate_num:
        noise = torch.randn(100, 32)
        if args.use_cuda:
            noise = noise.cuda()

        samples = netG(noise)
        samples = samples.view(100, 2, 32, 32)

        if args.sample_num - generate_num < 100:
            samples = samples[args.sample_num - generate_num]

        generate_num += 100
        samples_list.append(samples.cpu().detach().numpy())

    samples = np.concatenate(samples_list, axis=0)
    save_batch(samples, args.sample_dir)
