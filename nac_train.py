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
import time

import tensorflow as tf

import nac_net
import ast

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('arch', '{}',
                            """Architecture string.""")
tf.app.flags.DEFINE_string('archname', 'mxn',
                            """Architecture name.""")
tf.app.flags.DEFINE_string('initcell', '{}',
                            """Init cell architecture string.""")
tf.app.flags.DEFINE_string('classificationcell', '{}',
                            """Classification cell architecture string.""")
tf.app.flags.DEFINE_boolean('count_params', False,
                            """Whether to simulate model just for counting params""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                            """dataset to use, currently only cifar10 and imagenet supported""")
tf.app.flags.DEFINE_string('dataset_split_name', 'train',
                            """dataset to use, currently only cifar10 and imagenet supported""")
tf.app.flags.DEFINE_string('gpus', '[]',
                            """dataset to use, currently only cifar10 and imagenet supported""")
tf.app.flags.DEFINE_string('mode', '[]',
                            """mode: construct or oneshot, used to enable collection of featuremap stats for construction""")


def train(nn):
  """Train CIFAR-10 for a number of steps."""

  with tf.Graph().as_default():

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    global_step_init = -1
    if ckpt and ckpt.model_checkpoint_path:
       global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
       global_step = tf.Variable(global_step_init, name='global_step', dtype=tf.int64, trainable=False)
    else:
       global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = nn.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    arch = FLAGS.arch
    archname = FLAGS.archname
    initcell = FLAGS.initcell
    classificationcell = FLAGS.classificationcell
    mode = FLAGS.mode
    scope="Nacnet"
    is_training=True
    logits = nn.inference(images,
                     arch,
                     archname,
                     initcell,
                     classificationcell,
                     mode,
                     is_training,
                     scope)

    # Calculate loss.
    loss = nn.loss(logits, labels)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = nn.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = global_step_init
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    saver = tf.train.Saver()
    if FLAGS.count_params:
        #For counting parameters
        param_stats = tf.profiler.profile(tf.get_default_graph(),
                        options=tf.profiler.ProfileOptionBuilder()
                        .with_max_depth(2)
                        .with_accounted_types(['_trainable_variables'])
                        .select(['params'])
                        .build())
        #For counting flops
        param_stats = tf.profiler.profile(tf.get_default_graph(),
                        options=tf.profiler.ProfileOptionBuilder()
                        .with_max_depth(1)
                        .select(['float_ops']).build())

    else:
      with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=300,
        save_summaries_steps=100,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
          print ("Restoring existing model")
          saver.restore(mon_sess, ckpt.model_checkpoint_path)

        while not mon_sess.should_stop():
          mon_sess.run(train_op)

def multi_gpu_train(nn):
  """Train CIFAR-10 for a number of steps."""

  with tf.Graph().as_default():

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    global_step_init = -1
    if ckpt and ckpt.model_checkpoint_path:
       global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
       global_step = tf.Variable(global_step_init, name='global_step', dtype=tf.int64, trainable=False)
    else:
       global_step = tf.contrib.framework.get_or_create_global_step()

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (nac_net.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * nac_net.NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(nac_net.INITIAL_LEARNING_RATE, global_step, decay_steps, nac_net.LEARNING_RATE_DECAY_FACTOR, staircase=True)
    opt = tf.train.GradientDescentOptimizer(lr)
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = nn.distorted_inputs()
      batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity = 2 * len(FLAGS.gpus))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    arch = FLAGS.arch
    archname = FLAGS.archname
    initcell = FLAGS.initcell
    classificationcell = FLAGS.classificationcell
    mode = FLAGS.mode
    num_gpus = len(FLAGS.gpus)

    scope="Nacnet"
    is_training=True
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in FLAGS.gpus:
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    # Dequeues one batch for the GPU
                    image_batch, label_batch = batch_queue.dequeue()
                    logits = nn.inference(image_batch,
                                            arch,
                                            archname,
                                            initcell,
                                            classificationcell,
                                            mode,
                                            is_training,
                                            scope)
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss = nn.tower_loss(scope, logits, label_batch)
                    tf.get_variable_scope().reuse_variables()
                    # Retain the summaries from the final tower. TODO: not a nice way to use the last iteration of the loop
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

    grads = nn.average_gradients(tower_grads)

    summaries.append(tf.summary.scalar('learning_rate', lr))

    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
            nac_net.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = global_step_init
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    saver = tf.train.Saver()
    #with tf.contrib.tfprof.ProfileContext('/tmp/profiler/' + FLAGS.archname) as pctx:
    if True:
      with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=300,
        save_summaries_steps=100,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            allow_soft_placement=True)) as mon_sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
          print ("Restoring existing model")
          saver.restore(mon_sess, ckpt.model_checkpoint_path)

        tf.train.start_queue_runners(sess=mon_sess)

        while not mon_sess.should_stop():
          mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
  nn = nac_net.NacNet();
  nn.maybe_download_and_extract()
  print("main")
  if not tf.gfile.Exists(FLAGS.train_dir):
     tf.gfile.MakeDirs(FLAGS.train_dir)

  FLAGS.gpus = ast.literal_eval(FLAGS.gpus)
  if len(FLAGS.gpus) == 0:
      train(nn)
  else:
      multi_gpu_train(nn)


if __name__ == '__main__':
  tf.app.run()
