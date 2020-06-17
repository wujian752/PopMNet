from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from builtins import input
from mgn.solvers.trainer import get_trainer, trainer_map
from mgn.solvers.inferencer import inferencer_map
from dataset.structure import StructureDataset
from mgn.model import PlanModel
from mgn.module.criterion import SeqCriterion
from mgn.module.encoder_decoder import encoder_decoder_factory

import os
import torch
import shutil
import mgn.options as options


def train(args, parser):
    torch.manual_seed(1)
    if os.path.exists(args.save_dir):
        while True:
            choice = input('The save directory is existed, delete it [y/n]: ').lower()
            if choice == 'y':
                shutil.rmtree(args.save_dir)
                os.makedirs(args.save_dir)
                break
            elif choice == 'n':
                break
            else:
                print('Please choose yes or no.')
    else:
        os.makedirs(args.save_dir)

    encoder_decoder = encoder_decoder_factory(arch=args.encoder_decoder, condition_on_chord=args.chords,
                                              condition_on_plan=args.plan)
    criterion = SeqCriterion()
    if args.restore_file is None:
        model = PlanModel(input_size=encoder_decoder.input_size, encoder_decoder=encoder_decoder,
                          num_classes=encoder_decoder.num_classes, args=args)
        trainer = get_trainer(name=args.trainer, model=model, criterion=criterion, args=args)
    else:
        trainer, args = trainer_map[args.trainer].resume(args.restore_file, criterion, parser)

    dataset = StructureDataset(pickle_file=args.data, split_percentage=args.split, args=args,
                               logger=trainer.logger)
    dataset.set_encoder_decoder(encoder_decoder)
    if hasattr(dataset, 'preprocess'):
        dataset.preprocess()
    trainer.set_dataset(dataset)
    trainer.cuda()

    trainer.train_steps()


def inference(args, parser):
    if not os.path.exists(args.generate_dir):
        os.makedirs(args.generate_dir)

    if args.restore_file is None:
        raise ValueError('Restore file is required in inference mode.')
    else:
        inferencer, args = inferencer_map[args.trainer].resume(args.restore_file, parser)

    inferencer.logging_setting(name=os.path.join(args.generate_dir, 'generate.log'))

    encoder_decoder = encoder_decoder_factory(arch=args.encoder_decoder, condition_on_chord=args.chords,
                                              condition_on_plan=args.plan)

    dataset = StructureDataset(pickle_file=args.data, split_percentage=args.split, args=args,
                               logger=inferencer.logger)
    dataset.set_encoder_decoder(encoder_decoder)
    if hasattr(dataset, 'preprocess'):
        dataset.preprocess()
    inferencer.set_dataset(dataset)

    inferencer.cuda()
    inferencer.inference()


def main():
    parser = options.get_parser('Train or Inferencer.')
    options.add_generation_args(parser)
    options.add_model_args(parser)
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_dataset_args(parser)
    options.add_solver_args(parser)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.inference:
        inference(args, parser)
    else:
        train(args, parser)


if __name__ == '__main__':
    main()
