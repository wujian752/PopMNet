
from dataset import BaseDataset, Split

from torch.utils.data import DataLoader
from utils.structure import parse_melody

import torch


class PlanGraphDataset(BaseDataset):
    def __init__(self, pickle_file, split_percentage, args, logger, max_num_nodes=32, phase='Train'):
        super(PlanGraphDataset, self).__init__(pickle_file, split_percentage, args, logger)

        self.max_num_nodes = max_num_nodes

        self.train_structures = []
        self.eval_structures = []

        self.train_graphs = []
        self.eval_graphs = []

        self._phase = Split.All
        self._loaders = {}

    def preprocess(self):
        super(PlanGraphDataset, self).preprocess()

        self.train_structures = [parse_melody(ls.melody._events, 16) for ls in self.train_data]
        self.eval_structures = [parse_melody(ls.melody._events, 16) for ls in self.eval_data]

        self.train()
        for i in range(len(self.train_structures)):
            self.preload[Split.Train].append(self[i])
        self.eval()
        for i in range(len(self.eval_structures)):
            self.preload[Split.Test].append(self[i])

    def train(self):
        self._phase = Split.Train

    def eval(self):
        self._phase = Split.Test

    def generate(self):
        self._phase = Split.Generate

    def __len__(self):
        if self._phase == Split.Train:
            return len(self.train_structures)
        elif self._phase == Split.Test:
            return len(self.eval_structures)
        elif self._phase == Split.Generate:
            return len(self.eval_structures)
        elif self._phase == Split.All:
            return len(self.structures)

    def __getitem__(self, item):
        if self._phase == Split.Train:
            if len(self.preload[Split.Train]) > item:
                return self.preload[Split.Train][item]
            relations = self.train_structures[item]
        elif self._phase == Split.Test:
            if len(self.preload[Split.Test]) > item:
                return self.preload[Split.Test][item]
            relations = self.eval_structures[item]
        elif self._phase == Split.Generate:
            relations = self.eval_structures[item]
        else:
            raise ValueError('Unknown phase.')

        adj_matrix = torch.zeros((self.max_num_nodes, self.max_num_nodes)).float()
        edge_attribute = torch.zeros((2, self.max_num_nodes, self.max_num_nodes)).float()
        for i, relation in enumerate(relations):
            if relation.id > 0:
                adj_matrix[i, i - relation.offset] = 1
                edge_attribute[relation.id - 1, i, i - relation.offset] = 1

        return adj_matrix, edge_attribute

    def build_loader(self, batch_size=32, *args, **kwargs):
        if self._phase not in self._loaders.keys():
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=True, pin_memory=True,
                              *args, **kwargs)
        else:
            return self._loaders[self._phase]
