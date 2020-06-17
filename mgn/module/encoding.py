# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from magenta.music import encoder_decoder


class OneHotEncoding(encoder_decoder.OneHotEncoding):
    def __init__(self, length, default_event):
        self.length = length
        self._default_event = default_event

    @property
    def num_classes(self):
        return self.length

    @property
    def default_event(self):
        return self._default_event

    def encode_event(self, event):
        return event

    def decode_event(self, event):
        return event
