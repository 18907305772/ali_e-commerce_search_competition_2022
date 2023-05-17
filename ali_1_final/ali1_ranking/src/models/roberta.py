import collections

import tensorflow as tf

from src.models.bert import BertModel
from src.models.module import BasedModule, Linear, Dropout
from src.utils.config_util import BertConfig
from src.utils.tensor_util import ranking_loss, kl_divergence_loss


class RobertaModel(BertModel):
    def __init__(self, config: BertConfig, scope, add_pooling_layer=False):
        super().__init__(config, scope, add_pooling_layer)


class RobertaForPairwiseMatching(BasedModule):
    def __init__(self, config: BertConfig, margin=0.1):
        super().__init__()
        self.config = config
        self.margin = margin
        self.alpha = 2
        self.roberta = RobertaModel(config, add_pooling_layer=True, scope='bert')
        self.similarity = Linear(config.hidden_size, 1, scope='similarity')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, pos_input_ids, neg_input_ids, pos_token_type_ids, neg_token_type_ids, neg_doc_distance):
        neg_outputs = self.roberta.call(neg_input_ids, neg_token_type_ids)
        pos_outputs = self.roberta.call(pos_input_ids, pos_token_type_ids)
        neg_pooled_output = neg_outputs.pooled_output
        pos_pooled_output = pos_outputs.pooled_output
        neg_pooled_output = self.dropout.call(neg_pooled_output)
        pos_pooled_output = self.dropout.call(pos_pooled_output)
        neg_similarity = self.similarity.call(neg_pooled_output)
        pos_similarity = self.similarity.call(pos_pooled_output)
        mean_similarity = (neg_similarity + pos_similarity) / 2
        neg_similarity = neg_similarity - mean_similarity  # TODO: 是否有必要在训练时加入上述操作？
        pos_similarity = pos_similarity - mean_similarity
        neg_similarity = tf.nn.sigmoid(neg_similarity)
        pos_similarity = tf.nn.sigmoid(pos_similarity)
        dynamic_margin = self.margin * (neg_doc_distance ** self.alpha)
        loss = ranking_loss(pos_similarity, neg_similarity, dynamic_margin)
        logits = tf.concat([pos_similarity, neg_similarity], axis=-1)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=loss, logits=logits)


class RobertaForPairwiseMatchingPrediction(BasedModule):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=True, scope='bert')
        self.similarity = Linear(config.hidden_size, 1, scope='similarity')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids):
        outputs = self.roberta.call(input_ids, token_type_ids)
        pooled_output = outputs.pooled_output
        pooled_output = self.dropout.call(pooled_output)
        similarity = self.similarity.call(pooled_output)
        scores = tf.nn.sigmoid(similarity)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=None, logits=scores)


class RobertaForPairwiseMatchingWithRDrop(BasedModule):
    def __init__(self, config: BertConfig, margin=0.1):
        super().__init__()
        self.config = config
        self.margin = margin
        self.alpha = 2.0
        self.kl_alpha = 16.0
        self.roberta = RobertaModel(config, add_pooling_layer=True, scope='bert')
        self.similarity = Linear(config.hidden_size, 1, scope='similarity')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, pos_input_ids, neg_input_ids, pos_token_type_ids, neg_token_type_ids, neg_doc_distance):
        outputs1 = self._call(pos_input_ids, neg_input_ids, pos_token_type_ids, neg_token_type_ids, neg_doc_distance)
        outputs2 = self._call(pos_input_ids, neg_input_ids, pos_token_type_ids, neg_token_type_ids, neg_doc_distance)
        probs1 = tf.nn.softmax(outputs1.logits, axis=-1)
        probs2 = tf.nn.softmax(outputs2.logits, axis=-1)
        kl_loss = kl_divergence_loss(probs1, probs2)
        loss = self.kl_alpha * kl_loss + 0.5 * (outputs1.loss + outputs2.loss)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=loss, logits=outputs1.logits)

    def _call(self, pos_input_ids, neg_input_ids, pos_token_type_ids, neg_token_type_ids, neg_doc_distance):
        neg_pooled_output = self.roberta.call(neg_input_ids, neg_token_type_ids).pooled_output
        pos_pooled_output = self.roberta.call(pos_input_ids, pos_token_type_ids).pooled_output
        neg_pooled_output = self.dropout.call(neg_pooled_output)
        pos_pooled_output = self.dropout.call(pos_pooled_output)
        neg_similarity = self.similarity.call(neg_pooled_output)
        pos_similarity = self.similarity.call(pos_pooled_output)
        mean_similarity = (neg_similarity + pos_similarity) / 2
        neg_similarity = neg_similarity - mean_similarity
        pos_similarity = pos_similarity - mean_similarity
        neg_similarity = tf.nn.sigmoid(neg_similarity)
        pos_similarity = tf.nn.sigmoid(pos_similarity)
        dynamic_margin = self.margin * (neg_doc_distance ** self.alpha)
        loss = ranking_loss(pos_similarity, neg_similarity, dynamic_margin)
        logits = tf.concat([pos_similarity, neg_similarity], axis=-1)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=loss, logits=logits)


class RobertaForPairwiseMatchingSubmission(BasedModule):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=True, scope='bert')
        self.similarity = Linear(config.hidden_size, 1, scope='similarity')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, query_input_ids, doc_input_ids):
        input_ids, token_type_ids = self.get_concat_inputs(query_input_ids, doc_input_ids)
        outputs = self.roberta.call(input_ids, token_type_ids)
        pooled_output = outputs.pooled_output
        pooled_output = self.dropout.call(pooled_output)
        similarity = self.similarity.call(pooled_output)
        scores = tf.nn.sigmoid(similarity)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=None, logits=scores)

    def get_concat_inputs(self, query_input_ids, doc_input_ids):
        # first finds non-zeros elements, then takes these nonzero-elements' indices, finally expands dims
        nonzero_indices = tf.where(tf.greater(query_input_ids, 0))  # dim = (num_nonzero, input_ids.rank)
        broadcast_ones = tf.ones_like(tf.reduce_sum(doc_input_ids, axis=-1, keepdims=True))  # [b, 1]
        query_input_ids = broadcast_ones * tf.expand_dims(tf.gather_nd(query_input_ids, nonzero_indices), 0)
        query_token_type_ids = tf.zeros_like(query_input_ids, dtype=tf.int32)

        doc_zeros = tf.zeros_like(doc_input_ids, dtype=tf.int32)
        doc_ones = tf.ones_like(doc_input_ids, dtype=tf.int32)
        doc_token_type_ids = tf.where(tf.greater(doc_input_ids, 0), doc_ones, doc_zeros)

        input_ids = tf.slice(
            tf.concat([query_input_ids, doc_input_ids], axis=-1),
            [0, 0], [-1, self.config.seq_length])
        token_type_ids = tf.slice(
            tf.concat([query_token_type_ids, doc_token_type_ids], axis=-1),
            [0, 0], [-1, self.config.seq_length])
        return input_ids, token_type_ids


