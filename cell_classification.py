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
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

"""Classification cell"""
class Classification(Cell):
    def __init__(self, nn, cellidx):
        self.cellidx = cellidx
        self.cellname = "Classification"
        self.nn = nn
        Cell.__init__(self)

    def cell(self, inputs, arch, is_training):
        nscope = 'Cell_'+self.cellname+'_' + str(self.cellidx)
        net=inputs
        reuse = None
        print(nscope, inputs, [inputs.get_shape().as_list()])
        with tf.variable_scope(nscope, 'classification_block', [inputs], reuse=reuse) as scope:
            for layer in sorted(arch.keys()):
                for branch in sorted(arch[layer].keys()):
                    b = arch[layer][branch]
                    if b["block"] == "reduce_mean":
                        net = tf.reduce_mean(net, [1, 2]);
                    elif b["block"] == "flatten":
                        net = slim.flatten(net)
                    elif b["block"] == "fc":
                        outputs = b["outputs"]
                        net = slim.fully_connected(net, outputs)
                    elif b["block"] == "fc-final":
                        outputs = b["outputs"]
                        inputs = b["inputs"]
                        weights_initializer=trunc_normal(1/float(inputs))
                        biases_initializer=tf.zeros_initializer()
                        weights_regularizer=None
                        activation_fn=None
                        net = slim.fully_connected(net, outputs,
                                      biases_initializer=biases_initializer,
                                      weights_initializer=weights_initializer,
                                      weights_regularizer=None,
                                      activation_fn=None)
                    elif b["block"] == "dropout":
                        keep_prob = b["keep_prob"]
                        net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training)
                    else:
                        print ("Invalid block");
                        exit(-1);
        print(nscope, net, [net.get_shape().as_list()])
        return net

