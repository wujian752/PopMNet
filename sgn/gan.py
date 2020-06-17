# -*- coding: utf-8 -*-

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.dim = dim

        preprocess = nn.Sequential(
            nn.Linear(dim, 4 * 4 * 4 * dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(dim, 2, 10, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :8, :8]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.dim = dim

        main = nn.Sequential(
            nn.Conv2d(2, dim, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_dim, 4*4*4*dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*dim, 4*4*4*dim),
            nn.ReLU(True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*dim, 4*4*4*dim),
            nn.ReLU(True),
            # nn.Linear(4*4*4*dim, 4*4*4*dim),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*dim, 4*4*4*dim),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * dim, 1)

    def forward(self, input):
        input = input.view(-1, 2, 32, 32)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * self.dim)
        out = self.output(out)
        return out.view(-1)
