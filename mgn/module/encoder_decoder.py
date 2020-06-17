from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from magenta.music import MelodyOneHotEncoding, OneHotEventSequenceEncoderDecoder
from magenta.music import ConditionalEventSequenceEncoderDecoder
from mgn.module.encoding import OneHotEncoding
from mgn.config import melody_config


class OneHotEncoderDecoder(object):
    def __init__(self, name=None):
        self.encoder_decoder = OneHotEventSequenceEncoderDecoder(
            one_hot_encoding=MelodyOneHotEncoding(
                min_note=melody_config['min_note'],
                max_note=melody_config['max_note']
            )
        )

        if name is not None:
            self.encoder_decoder.name = name


encoder_decoder_map = {
    'onehot': OneHotEncoderDecoder,
}


def encoder_decoder_factory(arch, condition_on_chord=False, condition_on_plan=False):

    if arch in encoder_decoder_map.keys():
        structure_type_encoder = OneHotEventSequenceEncoderDecoder(
            one_hot_encoding=OneHotEncoding(
                length=3,
                default_event=2
            ),
        )
        structure_type_encoder.name = 'StructureTypeEncoderDecoder'

        melody_encoder_decoder = encoder_decoder_map[arch](name='MelodyEncoderDecoder').encoder_decoder
        encoder_decoder = ConditionalEventSequenceEncoderDecoder(
            control_encoder_decoder=structure_type_encoder,
            target_encoder_decoder=melody_encoder_decoder)
        return encoder_decoder
    else:
        raise KeyError('Unknown architecture {} for encoder_decoder. Available encoder: {}'.format(
            arch, ', '.join(encoder_decoder_map.keys())))
