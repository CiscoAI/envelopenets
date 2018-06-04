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

"""Initialization (Stem) cell"""
class Init(Cell):
    def __init__(self, cellidx, nn):
        self.cellidx = cellidx
        self.cellname = "Init"
        self.nn = nn
        Cell.__init__(self)

    def cell(self, inputs, arch, is_training):
        nscope = 'Cell_'+self.cellname+'_' + str(self.cellidx)
        reuse = None
        print(nscope, inputs, [inputs.get_shape().as_list()])
        with tf.variable_scope(nscope, 'initial_block', [inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                    net=inputs
                    layeridx=0;
                    for layer in sorted(arch.keys()):
                         cells = []
                         for branch in sorted(arch[layer].keys()):
                             b = arch[layer][branch]
                             if b["block"] == "conv2d":
                                 output_filters = int(b["outputs"])
                                 kernel_size = b["kernel_size"]
                                 if "stride" not in b.keys():
                                     stride = 1
                                 else:
                                     stride = b["stride"]
                                 cell = slim.conv2d(net, output_filters, kernel_size, stride=stride, padding='SAME'); #, scope=scope)
                             elif b["block"] == "max_pool":
                                 kernel_size = b["kernel_size"]
                                 cell = slim.max_pool2d(net, kernel_size, padding='SAME', stride=2) #, scope=scope)
                             elif b["block"] == "lrn":
                                 cell = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) #, name=scope)
                             elif b["block"] == "dropout":
                                 keep_prob = b["keep_prob"]
                                 cell = slim.dropout(net, keep_prob=keep_prob)
                             else:
                                 print ("Invalid block");
                                 exit(-1);
                             cells.append(cell)
                         net = tf.concat(cells, axis=-1)

                         layeridx+=1
        print(nscope, net, [net.get_shape().as_list()])
        return net

