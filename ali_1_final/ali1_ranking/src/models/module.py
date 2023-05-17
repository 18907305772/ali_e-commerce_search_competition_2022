import collections
import math
import os
from functools import wraps
from typing import Union

import tensorflow as tf
import tensorflow.nn as nn

from src.utils.ckpt_util import filter_compatible_params
from src.utils.debug_util import collector
from src.utils.tensor_util import create_initializer, reshape_to_matrix, reshape_from_matrix


class Placeholder(tf.Tensor):
    """ Just for annotation. """
    pass


def session_check(function):
    """
    Decorator for `tf.Session()` checking
    :param function: Function needs to be checked before call.
    """
    @wraps(function)
    def do_check_before_call(self, *args, **kwargs):
        if self.sess is None:
            raise RuntimeError(f'You should `model.compile` the model '
                               f'before you call the `{function.__name__}` op')
        return function(self, *args, **kwargs)
    return do_check_before_call


class BasedModule(object):
    def __init__(self):
        self.training = True
        self.sess = None
        self.loss = None
        self.logits = None
        self.optimizer = None
        self.sess = None
        self.placeholders = None

    def call(self, *args: tf.Tensor, **kwargs: tf.Tensor):
        raise NotImplementedError

    def compile(self, optimizer=None, training=True, **kwargs: Placeholder):
        """
        Call for this op after model's declaration and before model's training.
        :param training: whether training or evaluating.
        :param optimizer: optimizer.
        :param kwargs: tf.placeholder keyword arguments to sub-class's `forward` op.
        """
        outputs = self.call(**kwargs)
        self.register_placeholder(**kwargs)
        self.sess = tf.Session()
        self.loss = outputs.loss
        self.logits = outputs.logits
        if training:
            if optimizer is None:
                raise ValueError("You must pass an `optimizer` while mode of model is training.")
            self.optimizer = optimizer.minimize(self.loss)
        else:
            self.training = False
        self.sess.run(tf.global_variables_initializer())

    @session_check
    def save(self, save_path, name=None, step=None):
        """ Save model to save path """
        print("Saving model to %s......" % save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tvars = tf.trainable_variables()
        saver = tf.train.Saver(tvars)
        name = "model.ckpt" if name is None else f'{name}.ckpt'
        name = name if step is None else f'{name}-{str(step)}'
        save_path = os.path.join(save_path, name)
        saver.save(self.sess, save_path)
        print("Saving model complete!")
        return save_path

    @session_check
    def save_to_pb(self, save_path, inputs=None, outputs=None):
        """
        Save model to pb form.
        :param inputs: Dict: where keys are names of placeholders and
        values are `tf.placeholder`s.
        :param outputs: Dict: where keys are names of output tensor and
        values are `tf.Tensor`s.
        :param save_path: model saved directory.
        :return: save_path.
        """
        print("Saving model to %s......" % save_path)
        inputs = self.placeholders if inputs is None else inputs
        outputs = {'loss': self.loss, 'logits': self.logits} if outputs is None else outputs
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs=inputs, outputs=outputs)
        builder.add_meta_graph_and_variables(
            sess=self.sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        builder.save()
        print("Saving model complete!")
        return save_path

    @session_check
    def load(self, ckpt_file):
        """ Load model from checkpoint. """
        if ckpt_file is None:
            return
        print("Restore model from %s......" % ckpt_file)
        # Filtering compatible parameters
        rvars, unloaded = filter_compatible_params(ckpt_file)
        for name in unloaded:
            print(f"Warning : unloaded parameters `{name}`")
        saver = tf.train.Saver(rvars)
        saver.restore(self.sess, ckpt_file)
        print("Restoring model complete!")

    @session_check
    def train(self, **kwargs: Union[tuple, list, int]):
        """
        Training the model.
        :param kwargs: Keyword training data to feed the placeholders.
        Keyword must be corresponding to the names of placeholders.
        :return: namedtuple('Outputs', ['loss', 'logits'])
        """
        feed_dict = {}
        for name, placeholder in self.placeholders.items():
            feed_dict[placeholder] = kwargs.get(name)
        self.sess.run([self.optimizer], feed_dict=feed_dict)
        loss, logits = self.sess.run([self.loss, self.logits], feed_dict=feed_dict)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=loss, logits=logits)

    @session_check
    def evaluate(self, **kwargs: Union[tuple, list, int]):
        """
        Evaluating the model.
        :param kwargs: Keyword evaluating data to feed the placeholders.
        Keyword must be corresponding to the names of placeholders.
        :return: namedtuple('Outputs', ['loss', 'logits'])
        """
        feed_dict = {}
        for name, placeholder in self.placeholders.items():
            feed_dict[placeholder] = kwargs.get(name)
        logits = self.sess.run(self.logits, feed_dict=feed_dict)
        Outputs = collections.namedtuple('Outputs', ['logits'])
        return Outputs(logits=logits)

    @session_check
    def console(self, name, **kwargs: Union[tuple, list, int]):
        feed_dict = {}
        for name, placeholder in self.placeholders.items():
            feed_dict[placeholder] = kwargs.get(name)
        return collector.console(name, feed_dict=feed_dict, sess=self.sess)

    def register_placeholder(self, **kwargs: Placeholder):
        """
        Define and register placeholders for model.
        :param kwargs: keyword placeholders.
        """
        self.placeholders = {}
        for key, value in kwargs.items():
            if type(value) == tf.Tensor:
                self.placeholders[key] = value
            else:
                raise TypeError(f'Placeholder must be the type of `tf.Tensor`.')

    def __del__(self):
        if self.sess:
            self.sess.close()
            print("Session closed!")


class Linear(BasedModule):
    def __init__(self, in_features, out_features, scope):
        super().__init__()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.weight = tf.get_variable(
                name='kernel',
                shape=[in_features, out_features],
                dtype=tf.float32,
                initializer=create_initializer(1. / math.sqrt(in_features)))
            self.bias = tf.get_variable(
                name='bias',
                shape=[out_features],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

    def call(self, x):
        original_shape = x.shape
        x = reshape_to_matrix(x)
        x = x @ self.weight + self.bias
        x = reshape_from_matrix(x, original_shape)
        return x


class LayerNorm(BasedModule):
    def __init__(self, hidden_size, scope, axis=-1):
        super().__init__()
        self.axis = axis
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.weight = tf.get_variable(
                name='gamma',
                shape=[hidden_size],
                initializer=tf.ones_initializer())
            self.bias = tf.get_variable(
                name='beta',
                shape=[hidden_size],
                initializer=tf.zeros_initializer())

    def call(self, x):
        _x_m = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        _x_v = tf.reduce_mean((x - _x_m) ** 2, axis=self.axis, keepdims=True)
        return (x - _x_m) / tf.sqrt(_x_v + 1e-12) * self.weight + self.bias


class Dropout(BasedModule):
    def __init__(self, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob

    def call(self, x):
        return nn.dropout(x, 1.0 - self.dropout_prob) if self.training else x


class Embedding(BasedModule):
    def __init__(self, vocab_size, embedding_size, scope):
        super().__init__()
        self.weight = tf.get_variable(
            name=scope,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(1. / math.sqrt(embedding_size)))

    def call(self, input_ids):
        return nn.embedding_lookup(params=self.weight, ids=input_ids)
