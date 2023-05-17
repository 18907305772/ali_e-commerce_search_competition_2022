import re

import tensorflow as tf

from tensorflow.train import Optimizer


class Adam(Optimizer):
    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 use_locking=False,
                 name="Adam"):
        super(Adam, self).__init__(use_locking, name)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        def _get_variable_name(p_name):
            """Get the variable name from the tensor name."""
            match = re.match("^(.*):\\d+$", p_name)
            if match is not None:
                p_name = match.group(1)
            return p_name

        assignments = [global_step.assign(global_step + 1)] if global_step else []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = _get_variable_name(param.name)
            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            # Standard Adam update.
            next_m = (tf.multiply(self.beta1, m) + tf.multiply(1.0 - self.beta1, grad))
            next_v = (tf.multiply(self.beta2, v) + tf.multiply(1.0 - self.beta2, tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            update_with_lr = self.lr * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        tvars = tf.trainable_variables() if var_list is None else var_list
        grads = tf.gradients(loss, tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return self.apply_gradients(zip(grads, tvars), global_step=global_step)

    def _apply_dense(self, grad, var):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass
