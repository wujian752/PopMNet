from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset, DataLoader
from dataset import Split

import os
import pickle


class BaseDataset(Dataset):
    def __init__(self, pickle_file, split_percentage=0.9, args=None, logger=None):
        if not os.path.isfile(pickle_file):
            raise ValueError('The pickl file {} doesn\'t exist.'.format(pickle_file))
        self.pickle_file = pickle_file
        self.args = args

        # Read lead sheets from pickle file.
        self.data = None
        self.get_data()

        # Split dataset
        seg = int(len(self.data) * split_percentage)
        self.train_data = self.data[:seg]
        self.eval_data = self.data[seg:]

        # Set logger
        self.logger = logger

        self._phase = Split.Train
        self._loaders = {}
        self.preload = {
            Split.Train: [],
            Split.Test: [],
            Split.Generate: []
        }

    def get_data(self):
        with open(self.pickle_file, 'rb') as f:
            self.data = pickle.load(f)

    def set_encoder_decoder(self, encoder_decoder):
        self.encoder_decoder = encoder_decoder

    def squash_data(self, lead_sheet):
        lead_sheet.squash(min_note=48, max_note=84, transpose_to_key=0)
        return lead_sheet

    def preprocess(self, squash_data=True):
        self.logger.info('Preprocess dataset.')
        self.logger.info('Size of dataset: {} (Train) {} (Eval)'.format(
            len(self.train_data), len(self.eval_data)))

        train_data, eval_data = [], []
        if squash_data:
            for i in range(len(self.train_data)):
                try:
                    lead_sheet = self.squash_data(self.train_data[i])
                    train_data.append(lead_sheet)
                except Exception as exc:
                    self.logger.warning('The {} training data can\'t be preprocess because {}'.format(i, exc))
                    continue

            for i in range(len(self.eval_data)):
                try:
                    eval_data.append(self.squash_data(self.eval_data[i]))
                except Exception as exc:
                    self.logger.warning('The {} eval data can\'t be preprocess because {}'.format(i, exc))

        self.train_data = train_data
        self.eval_data = eval_data
        self.logger.info('Done.')
        self.logger.info('Size of dataset: {} (Train) {} (Eval)'.format(len(self.train_data), len(self.eval_data)))

    def train(self):
        self._phase = Split.Train

    def eval(self):
        self._phase = Split.Test

    def generate(self):
        self._phase = Split.Generate

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def build_loader(self, batch_size=32, *args, **kwargs):
        if self._phase not in self._loaders.keys():
            self._loaders[self._phase] = DataLoader(dataset=self, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                    collate_fn=self.collate, *args, **kwargs)
        return self._loaders[self._phase]

    def collate(self, batch):
        raise NotImplementedError
