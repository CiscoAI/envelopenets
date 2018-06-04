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


import tensorflow as tf
from cell import Cell

slim = tf.contrib.slim

""" Envelope Cell"""
class CellEnvelope(Cell):
    def __init__(self, cellidx, channelwidth, net, nn, filters, mode, outputs):
        self.cellidx = cellidx
        self.nn = nn
        self.mode = mode
        self.cellname = "Envelope"
        self.numbranches = 4;
        self.numbins = 100;
        self.batchsize = int(net.shape[0])
        numfilters = len(filters.keys())
        self.output_per_filter = outputs
        img_dims = int(net.shape[1])
        self.imagesize = [img_dims, img_dims]
        Cell.__init__(self)
        scope = 'Cell'+str(self.cellidx)
        if self.mode == "construct":
            with tf.variable_scope(scope, reuse=False):
                for branch in filters.keys():
                    with tf.variable_scope(branch, reuse=False):
                        self.init_stats();

    def cell(self, inputs, channelwidth, is_training=True, filters=None):
        """
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          By default use stride=1 and SAME padding
        """
        dropout_keep_prob=0.8
        nscope = 'Cell_'+self.cellname+'_' + str(self.cellidx)
        end_points = {}

        #Scope convention: Celltype_Cellidx/Branch_Branchidx/BLocktype_BlockIdx 
        scope='Cell'+str(self.cellidx)
        nets, means, variances = [], [], []
        print(nscope, inputs, [inputs.get_shape().as_list()])
        with tf.variable_scope(scope):
            for branch in sorted(filters.keys()):
                with tf.variable_scope(branch):
                    conv_h, conv_w = filters[branch][0], filters[branch][2]
                    inchannels = inputs.get_shape().as_list()[3]
                    outchannels = self.output_per_filter
                    if filters[branch].endswith("sep"):
                        net = slim.separable_conv2d(inputs, outchannels, [conv_h, conv_w], 1, normalizer_fn=slim.batch_norm)
                    else:
                        net = slim.conv2d(inputs, outchannels, [conv_h, conv_w], normalizer_fn=slim.batch_norm)
                if self.mode == "construct":
                    mean, variance, msss = self.calc_stats(net, branch)
                    net = tf.Print(net, [mean], message="Variance:" + scope + "/" + branch+":")
                    net = tf.Print(net, [variance], message="Mean:" + scope + "/" + branch+":")
                    net = tf.Print(net, [msss], message="MeanSSS:" + scope + "/" + branch+":")

                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout', is_training=is_training)
                nets.append(net)
            net = tf.concat(axis=3, values=nets)
        print(nscope, net, [net.get_shape().as_list()])
        return net, end_points

    def init_stats(self):
        size = [self.batchsize, self.imagesize[0], self.imagesize[1], self.output_per_filter]
        sumsquaredsamples = tf.contrib.framework.model_variable("sumsquaredsamples", size, initializer=tf.zeros_initializer);
        sumsamples= tf.contrib.framework.model_variable("sumsamples", size, initializer=tf.zeros_initializer);
        samplecount = tf.contrib.framework.model_variable("samplecount", [1], initializer=tf.zeros_initializer) 

    def calc_stats(self, inputs, scope):
        with tf.variable_scope(scope, reuse=True):
            size = [self.batchsize, self.imagesize[0], self.imagesize[1], self.output_per_filter]
            sumsquaredsamples = tf.get_variable("sumsquaredsamples", size)
            sumsamples= tf.get_variable("sumsamples", size)

            samplecount = tf.get_variable("samplecount", [1])
            tsamplecount = tf.add(samplecount, tf.to_float(tf.constant(1)))
            samplecount = samplecount.assign(tsamplecount)

            #input is N*H*W*C. We need to calcualte running variance over time (i.e over the N Images in this batch and in all batches.
            #Hence need to reduce across the N dimension
            sum_across_batch = tf.reduce_sum(inputs, axis=0)
            tsumsamples = tf.add(sumsamples, sum_across_batch);
            sumsamples = sumsamples.assign(tsumsamples)
            #sumsamples = tf.Print(sumsamples, [sumsamples], message="Sum Samples");

            squared_inputs = tf.square(inputs)
            squared_sum_across_batch = tf.reduce_sum(squared_inputs, axis=0)
            tsumsquaredsamples = tf.add(sumsquaredsamples, squared_sum_across_batch);
            sumsquaredsamples = sumsquaredsamples.assign(tsumsquaredsamples)

            msss = (1/samplecount) * (sumsquaredsamples);
            msss = tf.reduce_mean(msss)
            #msss = tf.Print(msss, [msss], message="MeanSSS:");

            mean = (1/samplecount) * (sumsamples);
            #mean across all elements of the featuremap
            mean = tf.reduce_mean(mean)
            #mean = tf.Print(mean, [mean], message="Mean:");

            variance = (1/samplecount) * (sumsquaredsamples - (tf.square(sumsamples)/samplecount))
            #mean across all elements of the featuremap
            variance = tf.reduce_mean(variance)
            #variance = tf.Print(variance, [variance], message="Variance:");
            return mean , variance, msss

    def init_entropy(self):
        bincount = tf.contrib.framework.model_variable("bincount", [self.numbins], initializer=tf.zeros_initializer);
        framemapsum = tf.contrib.framework.model_variable("framemapsum", [1], initializer=tf.zeros_initializer);
        framemapcount = tf.contrib.framework.model_variable("framemapcount", [1], initializer=tf.zeros_initializer)

    def calc_entropy(self, inputs, scope):
        with tf.variable_scope(scope, reuse=True):
            maxtensor = tf.to_float(tf.size(inputs));

            bincount = tf.get_variable("bincount", [self.numbins])
            framemapsum= tf.get_variable("framemapsum", [1])
            framemapcount = tf.get_variable("framemapcount", [1])
            inputs = tf.Print(inputs, [inputs, tf.shape(inputs)], message="Framemap:", summarize=100);
            bn = tf.to_int32(tf.floor((tf.reduce_sum(inputs)/maxtensor) * (self.numbins -1)))
            tbincount = tf.scatter_add(bincount, bn, tf.to_float(tf.constant(1)));
            bincount = bincount.assign(tbincount)
            bincount = tf.Print(bincount, [tf.count_nonzero(bincount)], message="Non zero bins count:");

            tframemapsum = tf.add(framemapsum, tf.reduce_sum(inputs))
            framemapsum = framemapsum.assign(tframemapsum)

            tframemapcount = tf.add(framemapcount, tf.to_float(tf.constant(1)))
            framemapcount = framemapcount.assign(tframemapcount)

            meanactivation = tf.divide(framemapsum, framemapcount)
            pbin = tf.divide(tf.to_float(bincount), tf.to_float(framemapcount))
            entropy = tf.multiply(pbin , tf.log(pbin))
            entropy = tf.where(tf.is_nan(entropy), tf.zeros_like(entropy), entropy)
            entropy = tf.reduce_sum(entropy)
            entropy = tf.Print(entropy, [entropy], message=": raw entropy: ")
            entropy = tf.multiply(entropy, tf.multiply(meanactivation, tf.constant(-1.0)))
            entropy = tf.Print(entropy, [scope, entropy], message=": scaled entropy: ")
            return entropy
