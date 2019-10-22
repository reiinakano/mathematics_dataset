# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of how to write generated questions to text files.

Given an output directory, this will create the following subdirectories:

*   train-easy
*   train-medium
*   train-hard
*   interpolate
*   extrapolate

and populate each of these directories with a text file for each of the module,
where the text file contains lines alternating between the question and the
answer.

Passing --train_split=False will create a single output directory 'train' for
training data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate
import six
from six.moves import range

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write output text')
flags.DEFINE_boolean('train_split', True,
                     'Whether to split training data by difficulty')
flags.mark_flag_as_required('output_dir')


def main(unused_argv):
  generate.init_modules(FLAGS.train_split)

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if os.path.exists(output_dir):
    logging.fatal('output dir %s already exists', output_dir)
  logging.info('Writing to %s', output_dir)
  os.makedirs(output_dir)

  for regime, flat_modules in six.iteritems(generate.filtered_modules):
    regime_dir = os.path.join(output_dir, regime)
    os.mkdir(regime_dir)
    per_module = generate.counts[regime]
    for module_name, module in six.iteritems(flat_modules):
      question_path = os.path.join(regime_dir, module_name + '.qu')
      answers_path = os.path.join(regime_dir, module_name + '.an')
      padded_answers_path = os.path.join(regime_dir, module_name + '.pa')
      with open(question_path, 'w') as question_text_file:
        with open(answers_path, 'w') as answers_text_file:
          with open(padded_answers_path, 'w') as padded_answers_text_file:
            for i in range(per_module):
              if i % 1000 == 0: print(i)
              problem, _ = generate.sample_from_module(module)
              question_text_file.write(tokenize(str(problem.question)) + '\n')
              answers_text_file.write(tokenize(str(problem.intermediate_steps)) + '\n')
              padded_answers_text_file.write(pad_out_intermediate_answers(tokenize(str(problem.intermediate_steps))) + '\n')
      logging.info('Written %s and %s', question_path, answers_path)


def tokenize(x: str):
  """Tokenize questions and intermediate steps in the way expected by pytorch/fairseq"""
  x = x.replace('\n', '@')  # newlines replaced by @
  x = x.replace(' ', '_')  # spaces replaced by _
  x = ' '.join(x)
  return x


def pad_out_intermediate_answers(x: str):
  """Replaces characters between = and @ with <pad>"""
  arr = x.split(" ")

  new_arr = []
  replacing = False
  for c in arr:
    if not replacing:
      if c == '=':
        replacing = True
        new_arr.append('=')
      else:
        new_arr.append(c)
    else:
      if c == '@':
        replacing=False
        new_arr.append('@')
      else:
        new_arr.append('<pad>')
  return " ".join(new_arr)


if __name__ == '__main__':
  app.run(main)
