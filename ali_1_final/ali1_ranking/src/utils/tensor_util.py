import tensorflow as tf
import tensorflow.nn as nn


def reshape_to_matrix(input_tensor):
    """ Reshape tensor to a rank 2 tensor """
    width = input_tensor.shape[-1]
    return tf.reshape(input_tensor, [-1, width])


def reshape_from_matrix(input_tensor, original_shape):
    """ Reshape tensor from 2D tensor to shape. """
    shapes = []
    for shape in original_shape[:-1]:
        shapes.append(-1 if shape.value is None else shape)
    shapes.append(input_tensor.shape[-1])
    return tf.reshape(input_tensor, shapes)


def create_initializer(initializer_range=1.0):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def create_input_mask(input_ids, dtype=tf.int32):
    """
    create mask according to input_tensor
    non-0 donates valid, 0 donates invalid
    input_tensor:
    [[2, 3, 1, 0, 0], [5, 0, 0, 0, 0]]
    result:
    [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]]
    :param dtype: data type.
    :param input_ids: 2D tensor
    :return: input_mask, same shape of input_tensor
    """
    tensor_mask = tf.sign(tf.abs(input_ids))
    return tf.cast(tensor_mask, dtype=dtype)


def create_attention_mask(input_ids):
    """ Create 3D attention mask from a 2D tensor mask.
    Args:
        input_ids: 2D Tensor of shape [batch_size, from_seq_length].
    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    assert input_ids.shape.ndims == 2
    input_mask = create_input_mask(input_ids)
    # `to_mask` = [batch_size, 1, to_seq_length]
    input_mask = tf.expand_dims(input_mask, axis=1)
    input_mask = tf.cast(input_mask, tf.int32)
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones_like(input_ids, dtype=tf.int32)
    broadcast_ones = tf.expand_dims(broadcast_ones, axis=-1)
    mask = broadcast_ones * input_mask
    return mask


def gelu(input_tensor):
    """ Gaussian Error Linear Unit. """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def cross_entropy_loss(logits, labels, weights=None):
    """
    Compute Cross-Entropy Loss.
    :param logits: [..., num_classes] tensor.
    :param labels: [...] tensor. LongTensor.
    Same shape with 0th - (last - 1)th of logits.
    :param weights: [...] tensor, where `1` donates validate and
     `0` donates invalidate. Same shape with 0th - (last - 1)th of logits.
    :return: The mean of all examples' loss.
    """
    if weights is None:
        weights = tf.ones_like(labels, dtype=tf.float32)
    weights = tf.cast(weights, dtype=tf.float32)
    weights = tf.reshape(weights, [-1])
    logits = tf.cast(logits, dtype=tf.float32)
    num_classes = int(logits.shape[-1])
    logits = reshape_to_matrix(logits)
    log_probs = nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, dtype=tf.int32)
    labels = tf.one_hot(labels, depth=num_classes)
    per_example_loss = - tf.reduce_sum(log_probs * labels, axis=-1)
    nrt = tf.reduce_sum(weights * per_example_loss)
    dnm = tf.reduce_sum(weights) + 1e-12
    loss = nrt / dnm
    return loss


def gather_indexes(sequence_tensor, positions):
    """
    Gathers the vectors at the specific positions over a mini-batch.
    :param sequence_tensor: [batch, seq_length, hidden_size]
    :param positions: [batch, max_predictions_pre_seq]
    :return: [batch, max_predictions_pre_seq, hidden_size]
    """
    assert len(sequence_tensor.shape) == 3
    assert len(positions.shape) == 2
    positions = tf.cast(positions, dtype=tf.int32)
    output_tensor = tf.batch_gather(sequence_tensor, positions)
    return output_tensor


def ranking_loss(positive, negative, margin):
    loss = tf.nn.relu(negative - positive + margin)
    loss = tf.reduce_mean(loss)
    return loss


def kl_divergence_loss(P, Q):
    """
    Compute average KL Divergence Loss:
    ```math:
    1/2 * [P * log(P/Q) + Q * log(Q/P)]
    ```
    :param P: Probability distribution. Tensor, same shape with Q;
    :param Q: Probability distribution. Tensor, same shape with P;
    :return: KL Divergence
    """
    divergence = 0.5 * (P * tf.log(P / Q) + Q * tf.log(Q / P))
    divergence = tf.reduce_sum(divergence, axis=-1)
    return tf.reduce_mean(divergence)
