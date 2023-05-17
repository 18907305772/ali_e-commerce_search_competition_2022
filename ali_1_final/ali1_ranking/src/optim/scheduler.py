import tensorflow as tf
from tensorflow.train import Optimizer


class Scheduler(object):
    def __init__(self, optimizer: Optimizer, total_steps: int):
        self.optimizer = optimizer
        self.total_steps = total_steps

    def apply(self, *args, **kwargs):
        """Apply scheduler strategy."""
        raise NotImplementedError


class PolynomialScheduler(Scheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int):
        super().__init__(optimizer, total_steps)

    def apply(self, power=1.0, end_learning_rate=0.0):
        """
        Apply polynomial learning rate decay strategy.
        :param power: power of the polynomial.
        :param end_learning_rate: the minimum end learning rate.
        :return: `Optimizer` object.
        """
        self.optimizer.lr = tf.train.polynomial_decay(
            learning_rate=self.optimizer.lr,
            global_step=self.optimizer.global_step,
            decay_steps=self.total_steps,
            end_learning_rate=end_learning_rate,
            power=power)
        return self.optimizer
