
from .base import BaseTrainer
from .plan import PlanTrainer

__all__ = [
    'trainer_map',
    'get_trainer',
    'BaseTrainer',
]

trainer_map = {
    'plan': PlanTrainer,
}


def get_trainer(name, *args, **kwargs):
    return trainer_map[name](*args, **kwargs)
