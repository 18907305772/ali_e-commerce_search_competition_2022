import collections
import re

import keras
import tensorflow as tf
from tensorflow.contrib.framework import load_variable, list_variables


def get_variable_name(variable):
    """
    Convert Variable object to it's name.
    :param variable: Variable object.
    :return: `str` name of Variable object.
    """
    name = variable.name
    match = re.match("^(.*):\\d+$", name)
    if match is not None:
        name = match.group(1)
    return name


def filter_compatible_params(checkpoint_file):
    """
    Filter the same parameters both in `trainable params` and `checkpoint params`.
    For example:
        in trainable params:
        <tf.Variable 'A:0' shape=(21128, 768) dtype=float32_ref>
        <tf.Variable 'B:0' shape=(512, 768) dtype=float32_ref>
        <tf.Variable 'C:0' shape=(768,) dtype=float32_ref>
        in checkpoint params:
        ('F', [21128, 768])
        ('B', [512, 768])
        ('D', [512, 768])
        ('C', [768])
        result:
        [<tf.Variable 'B:0' shape=(512, 768) dtype=float32_ref>,
        <tf.Variable 'C:0' shape=(768,) dtype=float32_ref>]
        Because only `C` and `B` are both shared by `trainable params` and `checkpoint params`
    :param checkpoint_file: directory of checkpoint file.
    :return: list of tf.Variable, list of str.
    """
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    ckpt_vars = reader.get_variable_to_shape_map()
    ckpt_vars = sorted(ckpt_vars.items(), key=lambda x: x[0])
    train_vars = tf.trainable_variables()
    loaded_vars = []
    unloaded_var_names = []
    for ckpt_var in ckpt_vars:

        if type(ckpt_var) is str:
            ckpt_var_name = ckpt_var
        elif type(ckpt_var) is tuple:
            ckpt_var_name = ckpt_var[0]
        else:
            raise ValueError("Unknown checkpoint type: %s" % type(ckpt_var))

        unloaded_var_names.append(ckpt_var_name)
        for train_var in train_vars:
            train_var_name = get_variable_name(train_var)
            if train_var_name == ckpt_var_name:
                loaded_vars.append(train_var)
                unloaded_var_names.pop()
                break
    return loaded_vars, unloaded_var_names


def print_param(name):
    """ Print the parameters in the graph. """
    with tf.Session() as sess:
        param = sess.graph.get_tensor_by_name(name)
        print(sess.run(param))


def print_checkpoint_variables(checkpoint_file):
    """ Print some variables from model checkpoint file. """
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    var_dict = reader.get_variable_to_shape_map()
    var_dict = sorted(var_dict.items(), key=lambda x: x[0])
    for item in var_dict:
        if 'adam' in item[0] or 'Adam' in item[0]:
            continue
        print(item)


def rename_checkpoint_variables(checkpoint_file, save_path="./result/model.ckpt"):
    """ Rename checkpoint variables """
    sess = tf.Session()
    name_to_variable = collections.OrderedDict()
    for var_name, _ in list_variables(checkpoint_file):
        var = load_variable(checkpoint_file, var_name)
        var_name = var_name.replace('s/0/s', 'bert/embeddings/word_embeddings')
        name_to_variable[var_name] = var
    saver = tf.train.Saver(name_to_variable)
    saver.save(sess, save_path)
    sess.close()


def restore_model(checkpoint_file):
    """ restore model params from checkpoint file. """

    def get_assignment_map():
        initialized_variable_names = {}
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = get_variable_name(var)
            name_to_variable[name] = var
        init_vars = tf.train.list_variables(checkpoint_file)
        assign_map = collections.OrderedDict()
        for x in init_vars:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assign_map[name] = name
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1
        return assign_map, initialized_variable_names

    tvars = tf.trainable_variables()
    print("Loading Trainable Variables From init_checkpoint: %s" % checkpoint_file)
    (assignment_map, _) = get_assignment_map()
    tf.train.init_from_checkpoint(checkpoint_file, assignment_map)

    # @classmethod
    # def get_assignment_map(cls, checkpoint):
    #     """
    #     Get model assignment map from checkpoint.
    #     :return: assignment_map: Dict, where keys are names of the parameters in the
    #     checkpoint and values are names of current parameters(in default graph).
    #     """
    #     assignment_map = collections.OrderedDict()
    #     named_parameters = cls.get_named_parameters()
    #     for name, param in tf.train.list_variables(checkpoint):
    #         if name not in named_parameters:
    #             continue
    #         assignment_map[name] = name
    #     return assignment_map
    #
    # @classmethod
    # def get_named_parameters(cls):
    #     """
    #     Get model named parameters.
    #     :return: named_parameters: Dict, where keys are names of the parameters in model,
    #     and values are parameters.
    #     """
    #     params = tf.trainable_variables()
    #     named_parameters = collections.OrderedDict()
    #     for param in params:
    #         match = re.match("^(.*):\\d+$", param.name)
    #         name = match.group(1) if match is not None else param.name
    #         named_parameters[name] = param
    #     return named_parameters
