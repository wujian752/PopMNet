from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn as nn
from mgn.config import model_config


rnn_types = {
    'RNN': nn.RNN,
    'GRU': nn.GRU,
    'LSTM': nn.LSTM,
}


class LanguageModel(nn.Module):
    def __init__(self, input_size, num_classes, encoder_decoder, condition_size=0,
                 embed_size=model_config['embed_size'],
                 hidden_size=model_config['hidden_size'], num_layers=model_config['num_layers'],
                 dropout_rate=model_config['dropout_rate'], args=None):
        super(LanguageModel, self).__init__()
        self.encoder_decoder = encoder_decoder

        self.args = args

        # Model
        self.input_size = input_size
        self.output_size = num_classes

        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(self.input_size, self.embed_size)

        if self.args.rnn == 'GRU':
            self.rnn = nn.GRU(input_size=self.embed_size + condition_size,
                              hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        elif self.args.rnn == 'RNN':
            self.rnn = nn.RNN(input_size=self.embed_size + condition_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=self.dropout_rate)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight = nn.init.xavier_uniform_(self.embedding.weight)
        # self.rnn.weight_hh_l0 = nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        # self.rnn.weight_ih_l0 = nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden.cuda()

    def forward(self, inputs, initial_states=None, full_output=False):
        """
        :param inputs: torch Tensor in size (L - 1, N, K)
        :param initial_states: None or tuple of states
            (torch.Tensor in size (num_layers, N, hidden_size) * 2)
        :param full_output:
            if True, output (logit, states)
            if False, output logit
        :return:
            Tensor in shape
                Batch x Length x output_size
        """
        # Variable in size of (L - 1, N, K)
        input_size = inputs.size()
        # Variable in size of ((L - 1) * N, K) for linear embedding.
        inp = inputs.view(-1, inputs.size(2))

        if initial_states is None:
            state = self.init_hidden(input_size[1])
        else:
            state = initial_states

        embeded = self.embedding(inp)
        # Variable in size of (L - 1, N, Embedding_Size).
        embeded = embeded.view(input_size[0], input_size[1], embeded.size(1))

        # Variable that contains the output features $h_t$ from the last layer of the RNN, for each t.
        # In size of (L, N, Hidden_Size)
        # Note that the $h_0$ initialized as zeros are also contained.
        outputs = [torch.zeros(embeded.size(1), self.hidden_size).cuda()]
        for i in range(input_size[0]):
            input_vector = embeded[i, :, :]
            output, state = self.rnn(input_vector.unsqueeze(0), state)
            outputs.append(output)

        outputs = torch.cat(outputs[1:], 0)
        output = outputs.view(-1, outputs.size(2))

        logit = self.linear(output)
        logit = logit.view(input_size[0], input_size[1], logit.size(1))

        return (logit, state) if full_output else logit

    def generate_steps(self, num_steps, primer_events, condition_sequences=None, state=None, full_output=False,
                       contain_primer=False, event_sequence=None):
        num_steps = min(num_steps, min([len(sequence) for sequence in condition_sequences]))

        if event_sequence is None:
            event_sequence = copy.deepcopy(primer_events)
        else:
            event_sequence = copy.deepcopy(event_sequence)

        inp = []
        for i in range(len(primer_events)):
            process_condition_sequences = self.condition_sequence_process(condition_sequences, event_sequence)
            inp.append(self.events_to_input(primer_events, process_condition_sequences, i))

        inputs = torch.from_numpy(np.array(inp)).unsqueeze(1)
        primer_events_var = inputs.cuda()

        for i in range(num_steps - len(primer_events)):
            logit, state = self.forward(primer_events_var.float(), initial_states=state, full_output=True)
            p = F.softmax(logit[-1], dim=1).data.cpu().squeeze().numpy()
            chosen_class = np.random.choice(p.shape[-1], p=p)
            event = self.encoder_decoder.class_index_to_event(chosen_class, event_sequence)
            event_sequence.append(event)

            if i == num_steps - len(primer_events) - 1:
                break

            process_condition_sequences = self.condition_sequence_process(condition_sequences, event_sequence)
            primer_events_var = torch.from_numpy(
                np.array(self.events_to_input(
                    event_sequence, process_condition_sequences, len(primer_events) + i))).cuda()
            primer_events_var = primer_events_var.unsqueeze(0).unsqueeze(0)

        if not contain_primer:
            event_sequence = event_sequence[-(num_steps - len(primer_events)):]

        return (event_sequence, state) if full_output else event_sequence

    def condition_sequence_process(self, condition_sequences, event_sequence):
        if not self.args.plan:
            return condition_sequences
        else:
            # Assume the offset is the last element in condition_sequences
            offset = condition_sequences[-1]
            condition_steps = [event_sequence[i - offset[i]] if offset[i] != 0 else 60
                               for i in range(len(event_sequence) + 1)]
            return condition_sequences[:-1] + [condition_steps]

    def events_to_input(self, target_sequence, condition_sequences, idx):
        return self.encoder_decoder.events_to_input(condition_sequences,
                                                    target_sequence,
                                                    position=idx)
