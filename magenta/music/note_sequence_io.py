# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""For reading/writing serialized NoteSequence protos to/from TFRecord files."""

import hashlib

# internal imports
from magenta.protobuf import music_pb2


def generate_note_sequence_id(filename, collection_name, source_type):
  """Generates a unique ID for a sequence.

  The format is:'/id/<type>/<collection name>/<hash>'.

  Args:
    filename: The string path to the source file relative to the root of the
        collection.
    collection_name: The collection from which the file comes.
    source_type: The source type as a string (e.g. "midi" or "abc").

  Returns:
    The generated sequence ID as a string.
  """
  # TODO(adarob): Replace with FarmHash when it becomes a part of TensorFlow.
  filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
  return '/id/%s/%s/%s' % (
      source_type.lower(), collection_name, filename_fingerprint.hexdigest())

