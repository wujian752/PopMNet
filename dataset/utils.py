
from enum import Enum, unique


@unique
class Split(Enum):
    Train = 0
    Test = 1
    Generate = 2
    All = 3
