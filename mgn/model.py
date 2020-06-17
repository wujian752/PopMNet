# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from magenta.music import PitchChordsEncoderDecoder

from mgn.base_model import LanguageModel
from mgn.config import melody_config
from mgn.config import model_config

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PlanModel(LanguageModel):
    def __init__(self, input_size, num_classes, encoder_decoder, args=None):
        self.chord_encoder_decoder = PitchChordsEncoderDecoder()
        self.condition_bar_encoder_decoder = encoder_decoder._target_encoder_decoder
        self.condition_bar_size = self.condition_bar_encoder_decoder.input_size
        self.condition_hidden_size = 128
        super(PlanModel, self).__init__(input_size,
                                        num_classes,
                                        encoder_decoder,
                                        condition_size=self.condition_hidden_size * 2,
                                        hidden_size=model_config['hidden_size'],
                                        num_layers=args.num_layers,
                                        args=args)

        self.condition_embedding = nn.Linear(self.condition_bar_size,
                                             self.embed_size)
        self.condition_rnn = nn.GRU(self.embed_size, self.condition_hidden_size // 2,
                                    num_layers=args.num_layers,
                                    dropout=0.5,
                                    bidirectional=True)
        self.chord_embedding = nn.Linear(self.chord_encoder_decoder.input_size,
                                         self.embed_size)
        self.chord_rnn = nn.GRU(self.embed_size, self.condition_hidden_size // 2,
                                num_layers=args.num_layers,
                                dropout=0.5,
                                bidirectional=True)

    def forward(self, inputs, condition_inputs, chord_inputs, initial_states=None, full_output=False):
        if condition_inputs.size(0) % melody_config['step_per_bar'] != 0:
            raise ValueError('Length of inputs {} is not consistant with steps_per_bar {}'.format(
                condition_inputs.size(0), melody_config['step_per_bar']))

        # Extract feature from control inputs
        condition_status = self.control_path(condition_inputs)
        chord_status = self.chord_path(chord_inputs)
        condition_status = condition_status[1:]
        chord_status = chord_status[1:]

        logit, state = self.target_path(inputs, torch.cat((condition_status, chord_status), dim=2))
        return (logit, state) if full_output else logit

    def target_path(self, target_inputs, control_status, states=None):
        input_size = target_inputs.size()

        # Emedding target inputs
        target_inputs = target_inputs.view(-1, target_inputs.size(-1))
        target_embeded = self.embedding(target_inputs)
        target_embeded = target_embeded.view(input_size[0], input_size[1], -1)

        merge_inputs = torch.cat((target_embeded, control_status), dim=2)

        if states is None:
            output, state = self.rnn(merge_inputs)
        else:
            output, state = self.rnn(merge_inputs, states)

        logit = self.linear(output)
        logit = logit.view(input_size[0], input_size[1], logit.size(2))

        return logit, state

    def control_path(self, control_inputs):
        input_size = control_inputs.size()

        if self.args.control_path != 'zero':
            control_inputs = control_inputs.view(-1, control_inputs.size(-1))
            condition_embeded = self.condition_embedding(control_inputs)
            condition_embeded = condition_embeded.view(input_size[0], input_size[1], condition_embeded.size(1))

            condition_output, h_n = self.condition_rnn(condition_embeded.view(melody_config['step_per_bar'],
                                                                              -1,
                                                                              condition_embeded.size(-1)))
            if self.args.control_path == 'full':
                condition_status = condition_output.view(input_size[0], input_size[1], -1)
            else:
                h_n = h_n.permute(1, 0, 2).contiguous()
                h_n = h_n.view(1, h_n.size(0), -1)
                h_n = h_n.expand(16, h_n.size(1), h_n.size(2))
                h_n = h_n.contiguous()
                condition_status = h_n.view(input_size[0], input_size[1], -1)

        return condition_status

    def chord_path(self, chord_inputs):
        if self.args.chord_path == 'zero':
            chord_states = torch.zeros((chord_inputs.size(0), chord_inputs.size(1),
                                        self.chord_rnn.hidden_size * 2)).float().cuda()
        else:
            chord_size = chord_inputs.size()
            chord_inputs = chord_inputs.view(-1, chord_inputs.size(-1))
            chord_embedding = self.chord_embedding(chord_inputs)
            chord_embedding = chord_embedding.view(chord_size[0], chord_size[1], chord_embedding.size(-1))
            chord_states, _ = self.chord_rnn(chord_embedding)
        return chord_states

    def generate_steps(self, num_steps, primer_events, condition_sequences=None, chord_progression=None,
                       relations=None, state=None, full_output=False, contain_primer=False, event_sequence=None):
        num_steps = min(num_steps, min([len(sequence) for sequence in condition_sequences]))
        num_steps = min(num_steps, len(chord_progression))

        structure, offset = condition_sequences

        if event_sequence is None:
            event_sequence = copy.deepcopy(primer_events)
        else:
            event_sequence = copy.deepcopy(event_sequence)

        chord_inputs = [self.chord_encoder_decoder.events_to_input(chord_progression, i)
                        for i in range(len(chord_progression))]
        chord_inputs = torch.FloatTensor(chord_inputs).unsqueeze(1).cuda()
        chord_status = self.chord_path(chord_inputs)

        condition_bar_idx = 0
        condition_status_tensor = torch.zeros((num_steps, 1, self.condition_hidden_size)).float().cuda()

        def full_condition_bar_tensor():
            condition_bar = self.condition_bar(event_sequence, offset, condition_bar_idx)
            condition_bar_tensor = torch.FloatTensor(condition_bar).cuda()
            condition_status_tensor[condition_bar_idx:condition_bar_idx + 16] = \
                self.control_path(condition_bar_tensor.unsqueeze(1))
            return condition_bar_idx + 16

        condition_bar_idx = full_condition_bar_tensor()

        inp = []
        for i in range(len(primer_events)):
            inp.append(self.events_to_input(primer_events, structure, i))
            if i == condition_bar_idx - 1:
                condition_bar_idx = full_condition_bar_tensor()

        target_inputs = torch.from_numpy(np.array(inp)).float().cuda().unsqueeze(1)
        control_inputs = condition_status_tensor[1:target_inputs.size(0) + 1]
        control_inputs = torch.cat((control_inputs, chord_status[1:target_inputs.size(0) + 1]), dim=2)

        for i in range(num_steps - len(primer_events)):
            logit, state = self.target_path(target_inputs, control_inputs, state)
            p = F.softmax(logit[-1], dim=1).data.cpu().squeeze().numpy()
            if self.args.sample_max:
                chosen_class = np.argmax(p)
            else:
                chosen_class = np.random.choice(p.shape[-1], p=p)
            event = self.encoder_decoder.class_index_to_event(chosen_class, event_sequence)

            event_sequence.append(event)

            if i == num_steps - len(primer_events) - 1:
                break

            if i + len(primer_events) == condition_bar_idx - 1:
                condition_bar_idx = full_condition_bar_tensor()

            # event_sequence = Melody(event_sequence)
            # event_sequence.transpose(0, min_note=48, max_note=84)
            # event_sequence = event_sequence._events
            target_inputs = torch.from_numpy(
                np.array(self.events_to_input(
                    event_sequence,
                    structure,
                    len(primer_events) + i))
            ).float().cuda()
            target_inputs = target_inputs.unsqueeze(0).unsqueeze(0)
            control_inputs = condition_status_tensor[len(primer_events) + i + 1:len(primer_events) + i + 2]
            control_inputs = torch.cat((control_inputs,
                                        chord_status[len(primer_events) + i + 1:len(primer_events) + i + 2]), dim=2)

        if not contain_primer:
            event_sequence = event_sequence[-(num_steps - len(primer_events)):]

        return (event_sequence, state) if full_output else event_sequence

    def condition_bar(self, event_sequence, offset, idx):
        start = idx // 16 * 16
        stop = start + 16

        condition_steps = []
        for i in range(start, stop):
            inputs = [0] * self.condition_bar_encoder_decoder.input_size
            if offset[i] != 0:
                inputs = self.condition_bar_encoder_decoder.events_to_input(event_sequence, i - offset[i])
            condition_steps.append(inputs)
        return condition_steps
