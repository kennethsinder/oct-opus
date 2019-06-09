""" A simple TensorFlow application"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf

# Create tensor
msg = tf.string_join(['Hello ', 'TensorFlow!'])

# Launch session
with tf.Session() as sess:
    print(sess.run(msg))

