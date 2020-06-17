# -*- coding: utf-8 -*-

from dataset import PlanGraphDataset
from argparse import ArgumentParser

import os

import logging
import coloredlogs

import tensorboardX

import torch
import torch.autograd as autograd
import torch.optim as optim

from sgn.gan import Generator
from sgn.gan import Discriminator

torch.manual_seed(1)


def logging_setting(name):
    # Define format
    fmt = '%(asctime)s, %(filename)s:%(lineno)d %(levelname)s %(message)s'
    # Add Notice Level
    logging.addLevelName(25, 'Notice')

    def notice(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)
    logging.Logger.notice = notice

    # Install Coloredlogs
    coloredlogs.install(level='INFO', fmt=fmt)

    # Get logger
    logger = logging.getLogger(__name__)
    # build and configure file handler
    file_handler = logging.FileHandler(name)
    file_handler.setLevel('INFO')
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    return logger


class DataIterator():
    def __init__(self, logger, args):
        self.dataset = PlanGraphDataset(args.data, 0.9, args=args,
                                        logger=logger)
        self.dataset.preprocess()
        self.dataset.train()
        self.data_loader = self.dataset.build_loader(batch_size=args.batch_size)

        self.reset()

    def reset(self):
        self.iterator = iter(self.data_loader)

    def next(self):
        try:
            data = self.iterator.next()
        except Exception as e:
            self.reset()
            data = self.iterator.next()
        return data[1]


def calc_gradient_penalty(netD, real_data, fake_data, lambda_wgan, use_cuda):
    # print real_data.size()
    alpha = torch.rand(real_data.size(0), 2, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_wgan
    return gradient_penalty


def wgan_train(netG, netD, optimizerG, optimizerD, data_iterator, args):
    one = torch.tensor(1.0)
    mone = one * -1
    if args.use_cuda:
        one = one.cuda()
        mone = mone.cuda()

    for iteration in range(args.iters):
        # (1) Update D network
        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(args.critic_iters):
            _data = data_iterator.next()
            real_data = torch.FloatTensor(_data)
            if args.use_cuda:
                real_data = real_data.cuda()
            real_data_v = autograd.Variable(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            # print D_real
            D_real.backward(mone)

            # train with fake
            noise = torch.randn((real_data_v.size(0), args.dim), requires_grad=False)
            if args.use_cuda:
                noise = noise.cuda()
            with torch.no_grad():
                fake_data = netG(noise).data
            D_fake = netD(torch.tensor(fake_data))
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake_data, args.lambda_wgan, args.use_cuda)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        noise = torch.randn(args.batch_size, args.dim)
        if args.use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            logger.info('[{}] D_cost: {:.4f} G_cost: {:.4f} W_D: {:.4f}'.format(
                iteration, D_cost.item(), G_cost.item(), Wasserstein_D.item()))
            logger.info('[{}] Real Mean: {:.4f} NumEdge: {:.2f}'.format(
                iteration, real_data_v.mean(), (real_data_v > 0.5).sum().float() / real_data_v.size(0)))
            logger.info('[{}] Fake Mean: {:.4f} Max: {:.4f} NumEdge: {:.2f}'.format(
                iteration, fake.mean(), fake.max(), (fake > 0.5).sum().float() / fake.size(0)))
            writer.add_scalar(tag='D_cost', scalar_value=D_cost.item(), global_step=iteration)
            writer.add_scalar(tag='G_cost', scalar_value=G_cost.item(), global_step=iteration)
            writer.add_scalar(tag='W_D', scalar_value=Wasserstein_D.item(), global_step=iteration)

        if (iteration + 1) % 1000 == 0:
            torch.save({'G': netG.state_dict(),
                        'D': netD.state_dict()},
                       '{}/{}.pth.tar'.format(os.path.join(args.dir, 'checkpoints'), iteration + 1))


if __name__ == '__main__':
    parser = ArgumentParser('Structure Generation Network.')
    parser.add_argument('--data', default='data/LS.pickle')
    parser.add_argument('--dir', default='results/sgn')
    parser.add_argument('--dim', default=32, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--critic_iters', default=5, type=int)
    parser.add_argument('--lambda_wgan', default=10, type=float)
    parser.add_argument('--iters', default=20000, type=int)
    parser.add_argument('--output_dim', default=32 * 32, type=int)
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
        os.makedirs(os.path.join(args.dir, 'checkpoints'))

    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.dir, 'train'))
    logger = logging_setting(name=os.path.join(args.dir, 'train.log'))
    netG = Generator(dim=args.dim)
    netD = Discriminator(dim=args.dim)

    if args.use_cuda:
        netD = netD.cuda()
        netG = netG.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    data_iterator = DataIterator(logger, args)

    wgan_train(netG, netD, optimizerG, optimizerD, data_iterator, args)
