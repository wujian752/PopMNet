from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from magenta.music import musicxml_reader
from magenta.music import note_sequence_io
from magenta.music import sequences_lib
from magenta.music import lead_sheets_lib
from magenta.music.sequences_lib import split_note_sequence_on_time_changes

import os
import traceback
import argparse
import logging
import coloredlogs
import pickle


parser = argparse.ArgumentParser('Convert Mxl files to json files.')

parser.add_argument('--input-dir', dest='input_dir', default='dataset', type=str,
                    help='folders where mxl files are stored in.')
parser.add_argument('--output-dir', dest='output_dir', default='data', type=str,
                    help='folder where json files are stored in.')


# Define format
fmt = '%(asctime)s, %(filename)s:%(lineno)d %(levelname)s %(message)s'
# Install Coloredlogs
coloredlogs.install(level='INFO', fmt=fmt)

logger = logging.getLogger(__name__)


class TimeSignatureException(Exception):
    pass


def convert_musicxml(root_dir, sub_dir, full_file_path):
    """Converts a musicxml file to a sequence proto.

    Copied from magenta/scripts/convert_dir_to_note_sequence.

    Args:
        root_dir: A string specifying the root directory for the files being
            converted.
        sub_dir: The directory being converted currently.
        full_file_path: the full path to the file to convert.

    Returns:
        Either a NoteSequence proto or None if the file could not be converted.
    """
    try:
        sequence = musicxml_reader.musicxml_file_to_sequence_proto(full_file_path)
    except musicxml_reader.MusicXMLConversionError as e:
        logger.warning(
            'Could not parse MusicXML file %s. It will be skipped. Error was: %s',
            full_file_path, e)
        return None
    reset_tempo_time(sequence)
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
    sequence.id = note_sequence_io.generate_note_sequence_id(
        sequence.filename, sequence.collection_name, 'musicxml')
    return sequence


def reset_tempo_time(ns):
    if len(ns.tempos) == 1 and ns.tempos[0].time > 0:
        ns.tempos[0].time = 0.0


def extract_lead_sheets(note_sequence):
    if len(note_sequence.time_signatures) > 1:
        raise TimeSignatureException('NoteSequence has more than 1 time signatures.')
    time_signature = note_sequence.time_signatures[0]
    if time_signature.denominator != 4 or time_signature.numerator != 4:
        raise TimeSignatureException('The time signature is not 4/4.')

    quantized_sequence = sequences_lib.quantize_note_sequence(note_sequence, steps_per_quarter=4)
    lead_sheets, _ = lead_sheets_lib.extract_lead_sheet_fragments(
        quantized_sequence,
        min_unique_pitches=5,
        gap_bars=20.0,
        ignore_polyphonic_notes=True,
        filter_drums=True,
        require_chords=True,
        pad_end=True,
        max_steps_truncate=512,
        all_transpositions=False
    )
    return lead_sheets


def preprocess(args):

    if not os.path.exists(args.input_dir):
        raise ValueError('Unknown path {}.'.format(args.input_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # build and configure file handler
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log'), mode='a')
    file_handler.setLevel('INFO')
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    note_sequences = []
    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.mxl') and not filename.endswith('.xml'):
            logger.warning('Unknown file type {}.'.format(filename))
            continue

        full_file_path = os.path.join(args.input_dir, filename)
        try:
            note_sequence = convert_musicxml(args.input_dir, '', full_file_path)
        except Exception as exc:
            logger.fatal('{} generated an exception: {}'.format(full_file_path, exc))
            traceback.print_exc()

        if note_sequence:
            note_sequences.extend(split_note_sequence_on_time_changes(note_sequence))

    lead_sheets_list = []
    for note_sequence in note_sequences:
        try:
            lead_sheets = extract_lead_sheets(note_sequence)
        except Exception as exc:
            logger.warning('Can\'t extract leadsheet from {} because {}'.format(
                note_sequence.filename, exc))
            continue
        lead_sheets_list.extend(lead_sheets)

    with open(os.path.join(args.output_dir, 'LS.pickle'), 'wb') as f:
        pickle.dump(lead_sheets_list, f)

    print('Finished.')


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess(args)
