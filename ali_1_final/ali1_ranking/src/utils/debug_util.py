import tensorflow as tf

from src.utils.visual_util import draw_matrix


class TensorCollector:

    def __init__(self):
        self._collections = {}
        self._sess = tf.Session()

    def collect(self, tensor, name):
        self._collections[name] = tensor

    def get(self, name):
        return self._collections.get(name, None)

    def initialize(self):
        self._sess.run(tf.global_variables_initializer())

    def console(self, name: str, feed_dict=None, sess=None):
        if name not in self._collections.keys():
            names = ' '.join([key for key in self._collections.keys()])
            raise ValueError('`%s` has not been collected yet!\n'
                             'collected [%s]' % (name, names))
        if sess is None:
            return self._sess.run(self._collections[name], feed_dict=feed_dict)
        else:
            return sess.run(self._collections[name], feed_dict=feed_dict)

    def run(self, tensor):
        return self._sess.run(tensor)

    def draw(self, tensor_name: str, title=''):
        tensor = self.console(tensor_name)
        draw_matrix(tensor, title=title)

    def __del__(self):
        self._sess.close()
        print("Session closed!")


collector = TensorCollector()
