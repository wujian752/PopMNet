from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import copy


class Relation(object):
    mode_ids = {'Create': 0, 'Repeat': 1, 'RhythmicSequence': 2}

    def __init__(self, mode, offset, info={}):
        self.mode = mode
        self.mode_id = Relation.mode_ids[self.mode]
        self.offset = offset
        self.infos = info

    @property
    def id(self):
        return self.mode_id

    def info(self, key):
        return self.info[key]


def parse_melody(melody, sub_module_len=16):
    melody = copy.deepcopy(melody)
    if melody[0] == -2:
        melody[0] = -1
    if len(melody) % sub_module_len != 0:
        raise ValueError()
    sub_melodies = [melody[i * sub_module_len:(i + 1) * sub_module_len]
                    for i in range(len(melody) // sub_module_len)]

    relations = []
    for i in range(len(sub_melodies)):
        relations.append(compare(sub_melodies[i], sub_melodies[:i]))
    return relations


def compare(source, targets):
    for i, target in enumerate(targets[::-1]):
        if equal(source, target):
            return Relation('Repeat', i + 1)
    for i, target in enumerate(targets[::-1]):
        if_seq = rhythmic_sequence(source, target)
        if if_seq:
            return Relation('RhythmicSequence', i + 1)
    return Relation('Create', 0)


def equal(a, b):
    return len(a) == len(b) and all([x == y for (x, y) in zip(a, b)])


def rhythmic_sequence(a, b):
    if len(a) != len(b):
        return False
    rhythm_a, pitch_a = divide_rhythm_and_pitch(a)
    rhythm_b, pitch_b = divide_rhythm_and_pitch(b)

    if equal(rhythm_a, rhythm_b):
        return True
    else:
        return False


def divide_rhythm_and_pitch(melody):
    rhythm, pitch = [], []
    for step in melody:
        if step == -2:
            rhythm.append(0)
        elif step == -1:
            rhythm.append(1)
            pitch.append(32)
        else:
            rhythm.append(1)
            pitch.append(step)
    return rhythm, pitch
