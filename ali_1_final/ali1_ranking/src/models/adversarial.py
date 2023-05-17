import re

import tensorflow as tf

from src.models.module import BasedModule


class FGM:
    """ Adversarial Training using Fast Gradient Method. """

    def __init__(self, model: BasedModule, epsilon=1.0, filter_scope='bert/embeddings'):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.filter_scope = filter_scope
        self.backup_grads = []
        params = tf.trainable_variables(self.filter_scope)

        # Attack
        grads = tf.gradients(self.model.loss, params)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        attack_assignments = []
        for grad, param in zip(grads, params):
            self.backup_grads.append(grad)
            grad_norm = tf.norm(grad)  # 将一个 batch 内所有样本计算L2范数
            next_param = param + tf.multiply(self.epsilon, grad) / grad_norm  # 梯度上升
            attack_assignments.append(param.assign(next_param))
        self.attack_op = tf.group(*attack_assignments)

        # Restore
        restore_assignments = []
        for grad, param in zip(self.backup_grads, params):
            grad_norm = tf.norm(grad)  # 将一个 batch 内所有样本计算L2范数
            next_param = param - tf.multiply(self.epsilon, grad) / grad_norm  # 梯度还原
            restore_assignments.append(param.assign(next_param))
        self.restore_op = tf.group(*restore_assignments)

    def attack(self, **kwargs):
        # Attack
        feed_dict = {}
        for name, placeholder in self.model.placeholders.items():
            feed_dict[placeholder] = kwargs.get(name)
        self.model.sess.run(self.attack_op, feed_dict=feed_dict)

    def restore(self, **kwargs):
        feed_dict = {}
        for name, placeholder in self.model.placeholders.items():
            feed_dict[placeholder] = kwargs.get(name)
        self.model.sess.run(self.restore_op, feed_dict=feed_dict)
