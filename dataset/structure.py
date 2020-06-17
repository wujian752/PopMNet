
from dataset import BaseDataset, Split

from torch.utils.data import DataLoader
from utils.structure import parse_melody
from magenta.music import PitchChordsEncoderDecoder

import torch


class StructureDataset(BaseDataset):
    def __init__(self, pickle_file, split_percentage, args, logger, phase='Train'):
        super(StructureDataset, self).__init__(pickle_file, split_percentage, args, logger)

        self.encoder_decoder = None
        self.chord_encoder_decoder = None
        self.condition_bar_encoder_decoder = None

        self._phase = Split.All
        self._loaders = {}

    def preprocess(self):
        super(StructureDataset, self).preprocess()

        self.train_structures = [parse_melody(ls.melody._events, 16) for ls in self.train_data]
        self.test_structures = [parse_melody(ls.melody._events, 16) for ls in self.eval_data]

        self.train()
        for i in range(len(self.train_data)):
            self.preload[Split.Train].append(self[i])
        self.eval()
        for i in range(len(self.eval_data)):
            self.preload[Split.Test].append(self[i])

    def set_encoder_decoder(self, encoder_decoder):
        self.encoder_decoder = encoder_decoder
        self.chord_encoder_decoder = PitchChordsEncoderDecoder()
        self.condition_bar_encoder_decoder = self.encoder_decoder._target_encoder_decoder

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
            return len(self.test_structures)
        elif self._phase == Split.Generate:
            return len(self.test_structures)
        elif self._phase == Split.All:
            return len(self.structures)

    def __getitem__(self, item):
        if self._phase == Split.Train:
            if len(self.preload[Split.Train]) > item:
                return self.preload[Split.Train][item]
            relations = self.train_structures[item]
            ls = self.train_data[item]
        elif self._phase == Split.Test:
            if len(self.preload[Split.Test]) > item:
                return self.preload[Split.Test][item]
            relations = self.test_structures[item]
            ls = self.eval_data[item]
        elif self._phase == Split.Generate:
            relations = self.test_structures[item]
            ls = self.eval_data[item]
        else:
            raise ValueError('Unknown phase.')

        structure = [relation.id for relation in relations for _ in range(16)]
        offset = [relation.offset * 16 for relation in relations for _ in range(16)]

        melody, chords = ls.melody._events, ls.chords._events

        if self._phase == Split.Generate:
            return ls, structure, offset, relations

        inp, condition = [], []

        condition_sequences = structure
        for i in range(len(melody) - 1):
            inp.append(self.events_to_input(melody,
                                            condition_sequences,
                                            i).unsqueeze(0))

        # Pad the melody so the the whole melody will be encoded.
        chord_condition = []
        for i in range(len(chords)):
            chord_condition.append(self.chord_encoder_decoder.events_to_input(chords, i))

        x = [self.condition_bar_encoder_decoder.events_to_input(melody, i) for i in range(len(melody))]
        for i in range(len(melody)):
            if offset[i] != 0:
                inputs = x[i - offset[i]]
            else:
                inputs = [0] * self.condition_bar_encoder_decoder.input_size
            condition.append(inputs)

        input = torch.cat(inp, dim=0)
        chord_input = torch.FloatTensor(chord_condition)
        condition_input = torch.FloatTensor(condition)
        structure = torch.LongTensor(structure)
        label = torch.LongTensor([self.encoder_decoder.events_to_label(melody, i) for i in range(1, len(melody))])
        return input, condition_input, chord_input, label, structure

    def build_loader(self, batch_size=32, *args, **kwargs):
        if self._phase not in self._loaders.keys():
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=self.collate, *args, **kwargs)
        else:
            return self._loaders[self._phase]

    def collate(self, batch):
        lengths = [input.size(0) for (input, _, _, _, _) in batch]
        max_length = max(lengths)

        inputs = torch.zeros((max_length, len(batch), batch[0][0].size(1)))
        condition_inputs = torch.zeros((max_length + 1, len(batch), batch[0][1].size(1)))
        chord_inputs = torch.zeros((max_length + 1, len(batch), batch[0][2].size(1)))
        labels = torch.zeros((max_length, len(batch))).long()
        masks = torch.zeros((max_length, len(batch)))
        structures = torch.zeros((max_length + 1, len(batch))).long()

        for i, (input, condition_input, chord_input, label, structure) in enumerate(batch):
            inputs[:lengths[i], i] = input
            condition_inputs[:lengths[i] + 1, i] = condition_input
            chord_inputs[:lengths[i] + 1, i] = chord_input
            labels[:lengths[i], i] = label
            masks[:lengths[i], i] = 1
            structures[:lengths[i] + 1, i] = structure

        return inputs, condition_inputs, chord_inputs, labels, masks, structures

    def events_to_input(self, sequence, condition_sequences, idx):
        inputs = self.encoder_decoder.events_to_input(condition_sequences,
                                                      sequence,
                                                      idx)
        return torch.FloatTensor(inputs)
