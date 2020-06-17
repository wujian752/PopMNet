from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch



def event_accuracy(output, target, mask, no_event_label):
    output = output.view(-1, output.size(2))
    target = target.view(-1)
    mask = mask.view(-1)

    event_positions = (1 - target.eq(no_event_label).float()) * mask
    no_event_positions = target.eq(no_event_label).float() * mask

    predictions = output.max(1)[1].type_as(target)
    correct = predictions.eq(target).float() * mask

    event_correct = correct * event_positions
    no_event_correct = correct * no_event_positions

    if event_positions.sum() > 0:
        event_acc = event_correct.sum() / event_positions.sum()
    else:
        event_acc = torch.FloatTensor([0.0])
    if no_event_positions.sum() > 0:
        no_event_acc = no_event_correct.sum() / no_event_positions.sum()
    else:
        no_event_acc = torch.FloatTensor([0.0])
    return event_acc, no_event_acc


def mode_accuracy(output, target, mask, mode, target_mode, no_event_label):
    if mode.size(0) != mask.size(0) + 1:
        if (mask.size(0) + 1) % mode.size(0) != 0:
            raise ValueError('The size of masks {} is not valid with size of mode {}'.format(
                mask.size(0), mode.size(0)
            ))
        expand_ratio = (mask.size(0) + 1) // mode.size(0)
        mode = mode.repeat(expand_ratio, 1)
    if mode.size(0) == mask.size(0) + 1:
        mode = mode[1:]

    mode = mode.view(-1)
    mode_mask = mode.eq(target_mode).float()

    output = output.view(-1, output.size(2))
    target = target.view(-1)
    mask = mask.view(-1)

    event_positions = (1 - target.eq(no_event_label).float()) * mask * mode_mask
    no_event_positions = target.eq(no_event_label).float() * mask * mode_mask

    predictions = output.max(1)[1].type_as(target)
    correct = predictions.eq(target).float() * mask * mode_mask

    event_correct = correct * event_positions
    no_event_correct = correct * no_event_positions

    if event_positions.sum() > 0:
        event_acc = event_correct.sum() / event_positions.sum()
    else:
        event_acc = torch.FloatTensor([0.0])
    if no_event_positions.sum() > 0:
        no_event_acc = no_event_correct.sum() / no_event_positions.sum()
    else:
        no_event_acc = torch.FloatTensor([0.0])
    return event_acc, no_event_acc


def accuracy(output, target, mask):
    output = output.view(-1, output.size(2))
    target = target.view(-1)
    mask = mask.view(-1)

    predictions = output.max(1)[1].type_as(target)
    correct = predictions.eq(target).float() * mask

    if correct.sum() > 0:
        acc = correct.sum() / mask.sum()
    else:
        acc = torch.FloatTensor([0.0])

    return acc
