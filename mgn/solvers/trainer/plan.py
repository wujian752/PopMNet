# -*- coding: utf-8 -*-

from solvers.trainer.base import BaseTrainer
from utils import accuracy, event_accuracy, mode_accuracy

import time


class PlanTrainer(BaseTrainer):
    def __init__(self, model, criterion, args):
        super(PlanTrainer, self).__init__(model, criterion, args)

    def define_record_infos(self):
        return ['loss', 'accuracy', 'event', 'noevent',
                '0_event', '0_noevent',
                '1_event', '1_noevent',
                '2_event', '2_noevent',
                ]

    def train_single_epoch(self, epoch, loader):
        for iteration, (inputs, condition_inputs, chord_inputs, label, mask, mode) in enumerate(loader):
            start_time = time.time()

            inputs, label, mask = inputs.cuda(), label.cuda(), mask.cuda()
            mode = mode.cuda()
            condition_inputs = condition_inputs.cuda()
            chord_inputs = chord_inputs.cuda()
            output = self.model(inputs, condition_inputs, chord_inputs)

            self.optimizer.zero_grad()

            loss = self.criterion(output, label, mask)
            loss.backward()
            self.optimizer.step()

            element_num = mask.sum().item()
            self.recorder['loss'].add(loss.item() * element_num, element_num)
            self.recorder['accuracy'].add(accuracy(output, label, mask) * element_num, element_num)
            event, noevent = event_accuracy(output, label, mask,
                                            no_event_label=self.dataset.encoder_decoder.default_event_label)
            self.recorder['event'].add(event.item() * element_num, element_num)
            self.recorder['noevent'].add(noevent.item() * element_num, element_num)
            for i in range(3):
                event, noevent = mode_accuracy(output, label, mask, mode, i,
                                               no_event_label=self.dataset.encoder_decoder.default_event_label)
                self.recorder['{}_event'.format(i)].add(event.item() * element_num, element_num)
                self.recorder['{}_noevent'.format(i)].add(noevent.item() * element_num, element_num)

            if iteration % self.args.print_freq == 0:
                self.record_infos(epoch, iteration, time.time() - start_time, writer=self.train_writer, phase='Train')

    def eval_single_epoch(self, epoch, loader, steps_of_epoch=0):
        for iteration, (inputs, condition_inputs, chord_inputs, label, mask, mode) in enumerate(loader):
            inputs, label, mask = inputs.cuda(), label.cuda(), mask.cuda()
            condition_inputs = condition_inputs.cuda()
            chord_inputs = chord_inputs.cuda()
            mode = mode.cuda()

            output = self.model(inputs, condition_inputs, chord_inputs)

            loss = self.criterion(output, label, mask)

            element_num = mask.sum()
            self.recorder['loss'].add(loss.item() * element_num, element_num)
            self.recorder['accuracy'].add(accuracy(output, label, mask) * element_num, element_num)
            event, noevent = event_accuracy(output, label, mask,
                                            no_event_label=self.dataset.encoder_decoder.default_event_label)
            self.recorder['event'].add(event.item() * element_num, element_num)
            self.recorder['noevent'].add(noevent.item() * element_num, element_num)
            for i in range(3):
                event, noevent = mode_accuracy(output, label, mask, mode, i,
                                               no_event_label=self.dataset.encoder_decoder.default_event_label)
                self.recorder['{}_event'.format(i)].add(event.item() * element_num, element_num)
                self.recorder['{}_noevent'.format(i)].add(noevent.item() * element_num, element_num)

        return_loss = self.recorder['loss'].value()[0]
        self.record_infos(epoch, steps_of_epoch, 0, self.test_writer, phase='Test')

        return return_loss
