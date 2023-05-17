import collections
import math

import tensorflow as tf
import tensorflow.nn as nn

from src.models.module import BasedModule, Linear, Dropout, LayerNorm, Embedding
from src.utils.config_util import TransformerConfig
from src.utils.debug_util import collector
from src.utils.tensor_util import reshape_to_matrix, gelu


class TransformerEmbeddings(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.word_embeddings = Embedding(
                config.vocab_size, config.hidden_size, scope='word_embeddings')
            self.position_embeddings = Embedding(
                config.max_position_embeddings, config.hidden_size, scope='position_embeddings')
            self.token_type_embeddings = Embedding(
                config.type_vocab_size, config.hidden_size, scope='token_type_embeddings')
            self.layer_norm = LayerNorm(config.hidden_size, scope='LayerNorm')
            self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids=None, position_ids=None):
        input_embeddings = self.word_embeddings.call(input_ids)
        if position_ids is None:
            position_ids = tf.constant([i for i in range(self.config.seq_length)], dtype=tf.int32)
        position_embeddings = self.position_embeddings.call(position_ids)
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)
        token_type_embeddings = self.token_type_embeddings.call(token_type_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm.call(embeddings)
        embeddings = self.dropout.call(embeddings)
        return embeddings


class TransformerSelfAttention(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.query = Linear(config.hidden_size, config.hidden_size, scope='query')
            self.key = Linear(config.hidden_size, config.hidden_size, scope='key')
            self.value = Linear(config.hidden_size, config.hidden_size, scope='value')
            self.dropout = Dropout(config.attention_probs_dropout_prob)
        assert config.hidden_size % config.num_attention_heads == 0
        self.size_per_head = int(config.hidden_size / config.num_attention_heads)
        self.attention_probs = None

    def call(self, hidden_states, attention_mask=None):
        reshaped_hidden_states = reshape_to_matrix(hidden_states)
        query = self.query.call(reshaped_hidden_states)
        key = self.key.call(reshaped_hidden_states)
        value = self.value.call(reshaped_hidden_states)
        query = tf.reshape(query, [-1, self.config.seq_length, self.config.num_attention_heads, self.size_per_head])
        key = tf.reshape(key, [-1, self.config.seq_length, self.config.num_attention_heads, self.size_per_head])
        value = tf.reshape(value, [-1, self.config.seq_length, self.config.num_attention_heads, self.size_per_head])
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        attention_scores = query @ tf.transpose(key, [0, 1, 3, 2])
        attention_scores = attention_scores * 1.0 / math.sqrt(self.size_per_head)

        if attention_mask is not None:
            expanded_attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(expanded_attention_mask, tf.float32)) * -10000.0
            attention_scores += adder

        attention_probs = nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout.call(attention_probs)
        attention_output = tf.matmul(attention_probs, value)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output, [-1, self.config.seq_length, self.config.num_attention_heads * self.size_per_head])
        return attention_output


class TransformerSelfAttentionOutput(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.dense = Linear(config.hidden_size, config.hidden_size, scope='dense')
            self.layer_norm = LayerNorm(config.hidden_size, scope='LayerNorm')
            self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, attention_output, input_tensor):
        attention_output = self.dense.call(attention_output)
        attention_output = self.dropout.call(attention_output)
        attention_output = self.layer_norm.call(attention_output + input_tensor)
        return attention_output


class TransformerAttention(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.self = TransformerSelfAttention(config, scope='self')
            self.output = TransformerSelfAttentionOutput(config, scope='output')

    def call(self, hidden_states, attention_mask=None):
        attention_output = self.self.call(hidden_states, attention_mask)
        attention_output = self.output.call(attention_output, hidden_states)
        return attention_output


class TransformerEncoderIntermediate(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.dense = Linear(config.hidden_size, config.intermediate_size, scope='dense')

    def call(self, hidden_states):
        hidden_states = self.dense.call(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class TransformerEncoderOutput(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.dense = Linear(config.intermediate_size, config.hidden_size, scope='dense')
            self.layer_norm = LayerNorm(config.hidden_size, scope='LayerNorm')
            self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense.call(hidden_states)
        hidden_states = self.dropout.call(hidden_states)
        hidden_states = self.layer_norm.call(hidden_states + input_tensor)
        collector.collect(hidden_states, 'e')
        return hidden_states


class TransformerEncoderLayer(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.attention = TransformerAttention(config, scope='attention')
            self.intermediate = TransformerEncoderIntermediate(config, scope='intermediate')
            self.output = TransformerEncoderOutput(config, scope='output')

    def call(self, hidden_states, attention_mask):
        attention_output = self.attention.call(hidden_states, attention_mask)
        intermediate_output = self.intermediate.call(attention_output)
        hidden_states = self.output.call(intermediate_output, attention_output)
        return hidden_states


class TransformerEncoder(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.layer = [TransformerEncoderLayer(config, scope=f'layer_{i}') for i in range(config.num_hidden_layers)]

    def call(self, embeddings_output, attention_mask=None):
        encoder_output = embeddings_output
        encoder_outputs = []
        for i in range(self.config.num_hidden_layers):
            encoder_output = self.layer[i].call(encoder_output, attention_mask)
            encoder_outputs.append(encoder_output)
        Outputs = collections.namedtuple('Outputs', ['encoder_outputs', 'final_encoder_output'])
        return Outputs(final_encoder_output=encoder_output, encoder_outputs=encoder_outputs)


class TransformerPooler(BasedModule):
    def __init__(self, config: TransformerConfig, scope):
        super().__init__()
        self.config = config
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.dense = Linear(config.hidden_size, config.hidden_size, scope='dense')

    def call(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense.call(first_token_tensor)
        pooled_output = tf.tanh(pooled_output)
        return pooled_output
