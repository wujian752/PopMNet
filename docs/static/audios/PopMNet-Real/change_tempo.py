# -*- coding: utf-8 -*-

import os
import pretty_midi


def change_tempo(filename, target_tempo):
    if not os.path.exists('120'):
        os.makedirs('120')
    origin_midi = pretty_midi.PrettyMIDI(filename)
    origin_tempo = origin_midi.get_tempo_changes()[1][0]

    new_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)

    for ins in origin_midi.instruments:
        new_ins = pretty_midi.Instrument(program=0)
        for note in ins.notes:
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start / target_tempo * origin_tempo,
                end=note.end / target_tempo * origin_tempo
            )
            new_ins.notes.append(new_note)
        new_midi.instruments.append(new_ins)
    new_midi.write('120/{}'.format(filename))

