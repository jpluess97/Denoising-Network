
from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import sparse_ops

# from tensorflow.contrib.nccl.ops import gen_nccl_ops
#from tensorflow.python.framework import add_model_variable

import tensorflow as tf
import numpy as np
import cv2

import warnings

def _relu(x, training=False, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name='relu'):
        return tf.nn.relu(x)

def _leak_relu(x, training=False, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name='lrelu'):
        return tf.nn.leaky_relu(x)
def _sigmoid(x, training=False, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name='sigmoid'):
        return tf.math.sigmoid(x)
def _htangent(x, training=False, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name='sigmoid'):
        return tf.math.tanh(x)




