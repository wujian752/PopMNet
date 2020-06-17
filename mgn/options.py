from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from module.encoder_decoder import encoder_decoder_map
from solvers.trainer import trainer_map

import argparse


def get_parser(desc):
    parser = argparse.ArgumentParser(
        description='RNN Music Generation Networks -- {}'.format(desc)
    )
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    return parser


def add_dataset_args(parser):
    group = parser.add_argument_group('')
    group.add_argument('data', metavar='DIR',
                       help='path to data directory.')
    group.add_argument('-s', '--split', default=0.9, type=float, metavar='SPLIT',
                       help='the split for training / generation.')
    group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                       help='number of data loading workers (default: 1)')
    group.add_argument('--shuffle', action='store_true')
    group.add_argument('--relations', type=str, default=None)
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--trainer', default='standard', type=str, metavar='STR', choices=trainer_map.keys(),
                       help='trainer type ({})'.format(', '.join(trainer_map.keys())))
    group.add_argument('--optimizer', default='Adam', type=str, metavar='STR',
                       help='optimizer type used to train model.')
    group.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR',
                       help='learning rate for initialization')
    group.add_argument('-b', '--batch-size', default=50, type=int, metavar='N',
                       help='batch size for training.')
    group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                       help='start training from this iteration.')
    group.add_argument('--max-epoch', '--me', default=100, type=int, metavar='N',
                       help='force stop training at specified iterations')
    group.add_argument('--clip-norm', default=25, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--weight-decay', '--wd', default=0.00001, type=float, metavar='WD',
                       help='weight decay')
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    group.add_argument('--restore-file', default=None, type=str,
                       help='filename in save-dir from which to load checkpoint')
    group.add_argument('--save-interval', type=int, default=1,
                       help='checkpoint every this many batches')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models and checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--resume', type=str)
    return group


def add_solver_args(parser):
    group = parser.add_argument_group('Solver')
    group.add_argument('--print-freq', dest='print_freq', metavar='N', default=10, type=int,
                       help='frequency of print training information.')
    return group


def add_generation_args(parser):
    group = parser.add_argument_group('Generation')
    group.add_argument('--inference', action='store_true',
                       help='Run in inference mode (True) or train mode (False).')
    group.add_argument('--generate-num', metavar='N', default=10, type=int,
                       help='num of melodies to generate.')
    group.add_argument('--generate-steps', metavar='N', default=512, type=int,
                       help='num of melodies to generate.')
    group.add_argument('--generate-dir', metavar='DIR', default='samples',
                       help='path to save samples.')
    group.add_argument('--primer-len', metavar='N', default=16, type=int,
                       help='length of primer used to initialize RNN.')
    group.add_argument('--conditional-chords', nargs='+',
                       help='The chords used as condition for melody generation.')
    group.add_argument('--qpm', metavar='INT', default=125, type=int,
                       help='Quarter notes per minute.')
    group.add_argument('--velocity', metavar='INT', default=100, type=int,
                       help='Midi velocity to give each note. Between 1 and 127 (inclusive).')
    group.add_argument('--instrument', metavar='INT', default=0, type=int,
                       help='Midi instrucment to give each note.')
    group.add_argument('--program', metavar='INT', default=0, type=int,
                       help='Midi program to give each note.')
    group.add_argument('--render-chords', action='store_true',
                       help='Whether if to render chords.')
    group.add_argument('--sample-max', action='store_true',
                       help='Whether if to render chords.')
    return group


def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')

    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    # Note: --arch cannot be combined with --encoder/decoder-* arguments.
    group.add_argument('--rnn', default='GRU', choices=['GRU', 'RNN', 'LSTM', 'INDRNN'],
                       help='the type of RNN used in model.')
    group.add_argument('--encoder-decoder', default='onehot', metavar='ARCH', choices=encoder_decoder_map.keys(),
                       help='discriminator architecture ({})'.format(', '.join(encoder_decoder_map.keys())))
    group.add_argument('--chords', action='store_true',
                       help='Train melody based on chords.')
    group.add_argument('--plan', action='store_true',
                       help='Train melody based on plan.')
    group.add_argument('--control-path', default='full', choices=['full', 'zero', 'final'],
                       help='control path type ({}).'.format(['full', 'zero', 'final']))
    group.add_argument('--chord-path', default='full', choices=['full', 'zero'],
                       help='control path type ({}).'.format(['full', 'zero']))
    group.add_argument('--num-layers', default=1, type=int)
    return group
