#Copyright 2018 Cisco Systems All Rights Reserved
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

#This code is derived from TensorFlow: https://github.com/tensorflow/models
#by The TensorFlow Authors, Google

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import nac_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('dataset_split_name', 'test',
                            """Classification cell architecture string.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('arch', '{}',
                            """Architecture string.""")
tf.app.flags.DEFINE_string('archname', 'mxn',
                            """Architecture name.""")
tf.app.flags.DEFINE_string('initcell', '{}',
                            """Init cell architecture string.""")
tf.app.flags.DEFINE_string('classificationcell', '{}',
                            """Classification cell architecture string.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                            """dataset to use, currently only cifar10 and imagenet supported""")
tf.app.flags.DEFINE_string('mode', '[]',
                            """mode: construct or oneshot, used to enable collection of featuremap stats for construction""")


def eval_once(saver, summary_writer, top_k_op, summary_op, k=1):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print(global_step)
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      print("total_sample_count is {} and num_examples is {}".format(total_sample_count, FLAGS.num_examples))
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      if k == 1:
          # Compute precision @ 1.
          precision = true_count / total_sample_count
          print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      elif k == 5:
          # Compute precision @ 5.
          precision = true_count / total_sample_count
          print('%s: precision @ 5 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ %d' % (k), simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(nn):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = nn.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #TODO: Clean up all args 
    arch = FLAGS.arch
    initcell = FLAGS.initcell
    classificationcell = FLAGS.classificationcell
    archname = FLAGS.archname
    mode = FLAGS.mode
    scope="Nacnet"
    is_training=False
    logits = nn.inference(images,
                     arch,
                     archname,
                     initcell,
                     classificationcell,
                     mode,
                     is_training,
                     scope)

    # Calculate predictions.
    #if imagenet is running then run precision@1,5
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    if FLAGS.dataset == "imagenet":
        #Quick dirty fixes to incorporate changes brought by imagenet
        FLAGS.num_examples = 50000
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        nac_net.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.dataset == "imagenet":
          eval_once(saver, summary_writer, top_5_op, summary_op, k=5)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  nn = nac_net.NacNet();
  nn.maybe_download_and_extract()
  if not tf.gfile.Exists(FLAGS.eval_dir):
     tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(nn)


if __name__ == '__main__':
  tf.app.run()
