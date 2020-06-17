# -*- coding: utf-8 -*-

import os
import json

from magenta.music import Melody, LeadSheet, ChordProgression, sequence_proto_to_midi_file
from magenta.music import BasicChordRenderer
from mgn.solvers.inferencer.base import BaseInferencer
from utils.structure import Relation


class PlanInferencer(BaseInferencer):
    def __init__(self, generator, dataset, args):
        super(PlanInferencer, self).__init__(generator, dataset, args)

    def single_generate(self, sample_idx, save_time, save_folder):
        ls, structure, offset, relations = self.dataset[sample_idx]
        melody_sequence, chords_sequence = ls.melody._events, ls.chords._events

        primer_sequence = melody_sequence[:self.args.primer_len]

        if self.args.conditional_chords is not None:
            chords_sequence = [chord for chord in self.args.conditional_chords for _ in range(16)]
        if self.args.relations is not None:
            with open(os.path.join(self.args.relations, '{}.json'.format(sample_idx))) as f:
                relations_list = json.load(f)
            default_relations = [Relation('Create', 0) for _ in range(32)]
            for relation in relations_list:
                default_relations[relation[1]].offset = relation[1] - relation[0]
                default_relations[relation[1]].mode = 'Repeat' if relation[2] == 0 else 'RhythmicSequence'
                default_relations[relation[1]].mode_id = Relation.mode_ids[default_relations[relation[1]].mode]
            relations = default_relations
            structure = [relation.id for relation in relations for _ in range(16)]
            offset = [relation.offset * 16 for relation in relations for _ in range(16)]

        condition_sequences = []
        chord_progression = chords_sequence

        condition_sequences.append(structure)
        condition_sequences.append(offset)

        self.generate(self.args.generate_steps, primer_sequence, condition_sequences, chord_progression, relations,
                      save_path=os.path.join(save_folder, '{}-{time}.mid'.format(sample_idx, time=save_time)))

    def generate(self, generate_steps, primer_sequence, condition_sequence, chord_progression, relations, save_path):
        note_sequence = self.generator.generate_steps(num_steps=generate_steps,
                                                      primer_events=primer_sequence,
                                                      condition_sequences=condition_sequence,
                                                      chord_progression=chord_progression,
                                                      relations=relations,
                                                      contain_primer=True)

        if self.args.chords:
            chords_sequence = chord_progression
            leadsheet = LeadSheet(Melody(note_sequence), ChordProgression(chords_sequence[:len(note_sequence)]))
            generated_sequence = leadsheet.to_sequence(velocity=self.args.velocity,
                                                       instrument=self.args.instrument,
                                                       qpm=self.args.qpm
                                                       )
            if self.args.render_chords:
                renderer = BasicChordRenderer()
                renderer.render(generated_sequence)
        else:
            melody = Melody(note_sequence)
            generated_sequence = melody.to_sequence(velocity=self.args.velocity,
                                                    instrument=self.args.instrument,
                                                    program=self.args.program,
                                                    qpm=self.args.qpm
                                                    )
        sequence_proto_to_midi_file(generated_sequence, save_path)
