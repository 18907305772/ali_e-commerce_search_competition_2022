import collections

import tensorflow as tf

from src.models.module import BasedModule, Linear, Dropout
from src.models.transformer import TransformerEncoder, TransformerEmbeddings, TransformerPooler
from src.utils.config_util import ErnieConfig
from src.utils.tensor_util import create_attention_mask, cross_entropy_loss, ranking_loss


class ErnieGramModel(BasedModule):
    def __init__(self, config: ErnieConfig, scope, add_pooling_layer=False):
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


class ErnieGramForSequenceClassification(BasedModule):
    def __init__(self, config: ErnieConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.ernie = ErnieGramModel(config, add_pooling_layer=True, scope='bert')
        self.classifier = Linear(config.hidden_size, num_classes, scope='classifier')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, labels, token_type_ids):
        outputs = self.ernie.call(input_ids, token_type_ids)
        pooled_output = outputs.pooled_output
        pooled_output = self.dropout.call(pooled_output)
        logits = self.classifier.call(pooled_output)
        predicts = tf.argmax(logits, axis=-1)
        predicts = tf.reshape(predicts, [-1])
        loss = cross_entropy_loss(logits, labels)
        Outputs = collections.namedtuple('Outputs', ['loss', 'predicts', 'logits', 'hidden_states'])
        return Outputs(loss=loss, predicts=predicts, logits=logits, hidden_states=pooled_output)


class ErnieGramForPairwiseMatching(BasedModule):
    def __init__(self, config: ErnieConfig, margin=0.1):
        super().__init__()
        self.config = config
        self.margin = margin
        self.ernie = ErnieGramModel(config, add_pooling_layer=True, scope='bert')
        self.similarity = Linear(config.hidden_size, 1, scope='similarity')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, pos_input_ids, neg_input_ids, pos_token_type_ids, neg_token_type_ids):
        neg_outputs = self.ernie.call(neg_input_ids, neg_token_type_ids)
        pos_outputs = self.ernie.call(pos_input_ids, pos_token_type_ids)
        neg_pooled_output = neg_outputs.pooled_output
        pos_pooled_output = pos_outputs.pooled_output
        neg_pooled_output = self.dropout.call(neg_pooled_output)
        pos_pooled_output = self.dropout.call(pos_pooled_output)
        neg_similarity = self.similarity.call(neg_pooled_output)
        pos_similarity = self.similarity.call(pos_pooled_output)
        neg_similarity = tf.nn.sigmoid(neg_similarity)
        pos_similarity = tf.nn.sigmoid(pos_similarity)
        logits = tf.concat([neg_similarity, pos_similarity], axis=-1)
        loss = ranking_loss(pos_similarity, neg_similarity, self.margin)
        Outputs = collections.namedtuple('Outputs', ['loss', 'logits'])
        return Outputs(loss=loss, logits=logits)
