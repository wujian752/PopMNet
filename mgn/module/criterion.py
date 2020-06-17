from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class SeqCriterion(nn.Module):
    def __init__(self):
        super(SeqCriterion, self).__init__()
        self.c = nn.NLLLoss(reduce=False)

    def forward(self, input, target, mask):
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        probs = F.log_softmax(input, dim=1)
        output = self.c(probs, target)

        assert output.size() == mask.size(), 'Size of output ({}) and mask ({}) are not match.'.format(
            output.size(), mask.size()
        )
        output = output * mask
        output = torch.sum(output) / torch.sum(mask)
        return output
