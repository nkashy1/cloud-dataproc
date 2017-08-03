# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tf import feature_column as fc

MODES = tf.estimator.ModeKeys


class Features(object):
  """
  Describes model features as TensorFlow feature columns based on preprocessing
  artifacts.
  """
  def __init__(self, artifact_dir):
    """
    Args:
      artifact_dir: Path to either local directory or GCS bucket containing
      preprocessing artifacts.
      
      This directory should contain one subdirectory
      for each of the integer features `integer-feature-$i` for i in {1..13}
      with each subdirectory having a `mean.txt` file defining the mean of that
      feature in the training data (conditioned on the feature being filled).

      It should also contain one subdirectory for each of the categorical
      features `categorical-feature-$i` for i in {1..26} with each subdirectory
      having two files - `count.txt` and `index.txt`. `index.txt` should contain
      the vocabulary items in decreasing order of frequency of appearance in the
      training data. `count.txt` should contain the number of lines in
      `index.txt`.

    Returns:
      None
    """
    self._artifact_dir = artifact_dir

    self._integer_features = ['integer-feature-{}'.format(i)
                              for i in range(1, 14)]

    self._categorical_features = ['categorical-feature-{}'.format(i)
                                  for i in range(1, 27)]

    self._inputs = {}

    for feature in self._integer_features:
      with tf.gfile.Open(self._artifact_dir + 'mean.txt', 'r') as mean_file:
        mean = float(mean_file.read())
      
      self._inputs[feature] = fc.numeric_column(key=feature,
                                                shape=(1,),
                                                default_value=mean,
                                                dtype=tf.int64)

    for feature in self._categorical_features:
      with tf.gfile.Open(self._artifact_dir + 'count.txt', 'r') as count_file:
        count = int(count_file.read())

      vocabulary_file = self._artifact_dir + 'index.txt'

      self._inputs[feature] = fc.categorical_column_with_vocabulary_file(
          key=feature,
          vocabulary_file=vocabulary_file,
          vocabulary_size=count,
          num_oov_buckets=1
      )

    self._label_key = 'clicked'
    self._label = {self._label_key: tf.numeric_column(key=self._label_key,
                                                      shape=(1,),
                                                      dtype=tf.int64)}

  def feature_spec(self, mode):
    """
    The features we expect to be present in a data set (based on whether that
    data set will be used for training, evaluation, or prediction.

    Args:
      mode: A tf.estimator.ModeKey - TRAIN, EVAL, or PREDICT

    Returns:
      A dictionary whose keys are feature labels and whose values are 
      tf.FixedLenFeature objects describing those features.
    """
    features = copy.copy(self._inputs)

    if mode in (MODES.TRAIN, MODES.EVAL):
      features[self._label_key] = self._label[self._label_key]

    return features


def generate_labelled_input_fn(batch_size, data_glob, features):
  """
  Args:
    batch_size: A positive integer specifying how large we would like each batch
    of training or evaluation to be
    data_glob: A glob which matches the tfrecords files containing labelled
    input data
    features: A Features object which provides a feature specification against
    which to decode tf.Example protos

  Returns:
    An input_fn which returns labelled data for use with tf.estimator.Estimator
  """
  feature_spec = features.feature_spec(MODES.TRAIN)

  def preprocessed_input_fn():
    input_batch = tf.contrib.learn.read_batch_features(
        file_pattern=data_glob,
        batch_size=batch_size,
        features=feature_spec,
        reader=tf.TFRecordReader,
        queue_capacity=20*batch_size,
        feature_queue_capacity=10*batch_size
    )

    label_batch = input_batch.pop("clicked")

    return input_batch, label_batch

  return preprocessed_input_fn


def generate_serving_input_receiver_fn(features):
  """
  Args:
    features: A Features object
  Returns:
    A serving_input_receiver_fn, which is a function of no arguments that
    returns a ServingInputReceiver, which specifies a serving signature for
    a given export of the estimator we are building
  """
  feature_spec = features.feature_spec(MODES.PREDICT)

  return tf.estimator.export.build_raw_serving_input_receiver_fn(
      features=feature_spec
  )


def generate_estimator():
  """
  Creates a TensorFlow linear classifier for use in Criteo prediction.

  Args:
    model_dir: The directory into which the classifer model should be stored
    for checkpoints.

  Returns:
    A tf.contrib.learn.LinearClassifier
  """
  feature_desc = get_features()
  feature_columns = [tf.feature_column.numeric_column(feature, dtype=tf.int64)
                     for feature in feature_desc]
  return tf.contrib.learn.LinearClassifier(feature_columns, model_dir)


def make_experiment_fn(estimator, train_glob, eval_glob, batch_size,
                       train_steps):
  def experiment_fn(*args, **kwargs):
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(batch_size, train_glob),
        eval_input_fn=make_input_fn(batch_size, eval_glob),
        train_steps=train_steps,
        export_strategies=(
            tf.contrib.learn.make_export_strategy(make_serving_fn())
        )
      )

  return experiment_fn


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
      "Spark Preprocessing + TensorFlow Canned Linear Classifier + Criteo data = <3"
  )
  parser.add_argument(
      "--job-dir",
      required=True,
      help=
      "The directory in which trained models should (and may already be) saved (can be a GCS path)"
  )
  parser.add_argument(
      "--train-dir",
      required=True,
      help=
      "Directory containing the training data")
  parser.add_argument(
      "--eval-dir",
      required=True,
      help=
      "Directory containing the evaluation data"
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=10000,
      help="The size of the batches in which the criteo data should be processed"
  )
  parser.add_argument(
      "--train-steps",
      type=int,
      help="The number of batches that we should train on (if unspecified, trains forever)"
  )

  args = parser.parse_args()
  
  train_glob = '{}*'.format(args.train_dir)
  eval_glob = '{}*'.format(args.eval_dir)

  estimator = make_estimator(args.job_dir)
  experiment_fn = make_experiment_fn(estimator, train_glob, eval_glob,
                                     args.batch_size, args.train_steps)

  tf.contrib.learn.learn_runner.run(experiment_fn, args.job_dir)
