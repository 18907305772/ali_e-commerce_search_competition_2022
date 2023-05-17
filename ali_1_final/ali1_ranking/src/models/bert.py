import collections

import tensorflow as tf

from src.models.module import BasedModule, Dropout, Linear, LayerNorm
from src.models.transformer import TransformerEmbeddings, TransformerPooler, TransformerEncoder
from src.utils.config_util import BertConfig
from src.utils.tensor_util import create_attention_mask, cross_entropy_loss, gelu, gather_indexes


class BertModel(BasedModule):
    def __init__(self, config: BertConfig, scope, add_pooling_layer=False):
        super().__init__()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.embeddings = TransformerEmbeddings(config, scope='embeddings')
            self.encoder = TransformerEncoder(config, scope='encoder')
            self.pooler = TransformerPooler(config, scope='pooler') if add_pooling_layer else None

    def call(self, input_ids, token_type_ids=None, position_ids=None):
        embedding_output = self.embeddings.call(input_ids, token_type_ids, position_ids)
        attention_mask = create_attention_mask(input_ids)
        encoder_outputs = self.encoder.call(embedding_output, attention_mask)
        pooled_output = self.pooler.call(encoder_outputs.final_encoder_output) if self.pooler else None
        Outputs = collections.namedtuple('Outputs', ['pooled_output', 'hidden_states'])
        return Outputs(pooled_output=pooled_output, hidden_states=encoder_outputs.final_encoder_output)


class BertPredictionHeadTransform(BasedModule):
    def __init__(self, config: BertConfig, scope):
        super().__init__()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.dense = Linear(config.hidden_size, config.hidden_size, scope='dense')
            self.layer_norm = LayerNorm(config.hidden_size, scope='LayerNorm')

    def call(self, hidden_states):
        hidden_states = self.dense.call(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.layer_norm.call(hidden_states)
        return hidden_states


class BertLMPredictionHead(BasedModule):
    def __init__(self, config: BertConfig, embedding_table, scope):
        super().__init__()
        self.config = config
        self.embedding_table = embedding_table
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.transform = BertPredictionHeadTransform(config, scope='transform')
            self.output_bias = self.bias = tf.get_variable(
                name='output_bias',
                shape=[config.vocab_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

    def call(self, hidden_states):
        hidden_states = self.transform.call(hidden_states)
        hidden_states = tf.matmul(hidden_states, self.embedding_table, transpose_b=True)
        hidden_states = tf.nn.bias_add(hidden_states, self.output_bias)
        return hidden_states


class BertForMaskLMHead(BasedModule):
    def __init__(self, config: BertConfig, embedding_table, scope):
        super().__init__()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.predictions = BertLMPredictionHead(config, embedding_table, scope='predictions')

    def call(self, hidden_states, positions):
        hidden_states = gather_indexes(hidden_states, positions)
        hidden_states = self.predictions.call(hidden_states)
        return hidden_states


class BertForMaskLM(BasedModule):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.input_ids = tf.placeholder(
            shape=[None, config.seq_length], dtype=tf.int32)
        self.bert = BertModel(config, add_pooling_layer=True, scope='bert')
        self.cls = BertForMaskLMHead(config, self.bert.embeddings.word_embeddings.weight, scope='cls')

    def call(self, input_ids, labels, positions, weights):
        outputs = self.bert.call(input_ids)
        logits = self.cls.call(outputs.hidden_states, positions)
        predicts = tf.argmax(logits, axis=-1)
        predicts = predicts * tf.sign(positions)
        loss = cross_entropy_loss(logits, labels, weights)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits', 'hidden_states', 'predicts'])
        return Outputs(loss=loss, logits=logits, hidden_states=outputs.hidden_states, predicts=predicts)


class BertForSequenceClassification(BasedModule):
    def __init__(self, config: BertConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=True, scope='bert')
        self.classifier = Linear(config.hidden_size, num_classes, scope='classifier')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, labels, token_type_ids):
        outputs = self.bert.call(input_ids, token_type_ids)
        pooled_output = outputs.pooled_output
        pooled_output = self.dropout.call(pooled_output)
        logits = self.classifier.call(pooled_output)
        predicts = tf.argmax(logits, axis=-1)
        predicts = tf.reshape(predicts, [-1])
        loss = cross_entropy_loss(logits, labels)
        Outputs = collections.namedtuple('Outputs', ['loss', 'predicts', 'logits', 'hidden_states'])
        return Outputs(loss=loss, predicts=predicts, logits=logits, hidden_states=pooled_output)
