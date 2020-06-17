import os
import random
import time
import logging
import coloredlogs

from utils.checkpoint import Checkpoint
from mgn.module.encoder_decoder import encoder_decoder_factory
from mgn.model import PlanModel


class BaseInferencer(object):
    def __init__(self, generator, dataset, args):
        self.generator = generator
        self.dataset = dataset
        self.args = args

    def set_dataset(self, dataset):
        self.dataset = dataset

    def cuda(self):
        self.generator.cuda()

    def logging_setting(self, name):
        # Define format
        fmt = '%(asctime)s, %(filename)s:%(lineno)d %(levelname)s %(message)s'

        # Install Coloredlogs
        coloredlogs.install(level='INFO', fmt=fmt)

        # Get logger
        self.logger = logging.getLogger(__name__)
        # build and configure file handler
        file_handler = logging.FileHandler(name)
        file_handler.setLevel('INFO')
        file_handler.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(file_handler)

    @classmethod
    def resume(cls, ckpt, parser):
        if os.path.islink(ckpt):
            checkpoint = Checkpoint.load(path=ckpt)
        else:
            checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(ckpt))

        args = checkpoint.args
        args = parser.parse_args(namespace=args)
        encoder_decoder = encoder_decoder_factory(arch=args.encoder_decoder, condition_on_chord=args.chords,
                                                  condition_on_plan=args.plan)

        model = PlanModel(input_size=encoder_decoder.input_size, encoder_decoder=encoder_decoder,
                          num_classes=encoder_decoder.num_classes, args=args)

        model.load_state_dict(checkpoint.model)

        inferencer = cls(model, dataset=None, args=args)
        return inferencer, args

    def __str__(self):
        output = self.generator.__str__()
        return output

    def inference(self, save_name=None):
        if self.args.generate_dir == 'samples':
            save_folder = os.path.join(self.args.save_dir, 'samples/{}'.format(save_name))
        else:
            save_folder = os.path.join(self.args.generate_dir)
        save_time = time.time()
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        self.dataset.generate()
        sample_idx = range(len(self.dataset))
        if self.args.shuffle:
            random.shuffle(sample_idx)

        self.generator.eval()
        for i in range(self.args.generate_num):
            self.single_generate(sample_idx[i], save_time, save_folder)

    def single_generate(self, sample_idx, save_time, save_folder):
        raise NotImplementedError
