from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from solvers.inferencer.base import BaseInferencer
from solvers.inferencer.plan import PlanInferencer


__all__ = [
    'inferencer_map',
    'get_inferencer',
    'BaseInferencer',
]


inferencer_map = {
    'plan': PlanInferencer,
}


def get_inferencer(name, *args, **kwargs):
    return inferencer_map[name](*args, **kwargs)
