import collections
import random


class MaskedLMDataCreator:
    def __init__(self,
                 tokenizer,
                 seq_length,
                 max_predictions=20,
                 masked_lm_prob=0.15):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions

    def create_test_example(self, text):
        """ Create data for mlm from single text. """
        _tokens = [self.tokenizer.CLS_TOKEN]
        _tokens.extend(self.tokenizer.tokenize(text))
        _labels, _positions, _weights = [], [], []
        for _i, token in enumerate(_tokens):
            if token.lower() == self.tokenizer.MASK_TOKEN.lower():
                _labels.append(self.tokenizer.MASK_TOKEN)
                _positions.append(_i)
                _weights.append(1)
        while len(_labels) < self.max_predictions_per_seq:
            _labels.append(self.tokenizer.PAD_TOKEN)
            _positions.append(0)
            _weights.append(0)
        _tokens.append(self.tokenizer.SEP_TOKEN)
        while len(_tokens) < self.seq_length:
            _tokens.append(self.tokenizer.PAD_TOKEN)
        if len(_tokens) > self.seq_length:
            _tokens = _tokens[0: self.seq_length - 1]
            _tokens.append(self.tokenizer.SEP_TOKEN)
        assert len(_tokens) == self.seq_length
        MaskedLmExample = collections.namedtuple(
            "MaskedLmExample", ["token_ids", "label_ids", "positions", "weights"])
        example = MaskedLmExample(
            token_ids=self.tokenizer.tokens2ids(_tokens),
            label_ids=self.tokenizer.tokens2ids(_labels),
            positions=_positions, weights=_weights)
        return example

    def create_example(self, text):
        """Creates the predictions for the masked LM objective."""
        MaskedLmInstance = collections.namedtuple(
            "MaskedLmInstance", ["index", "label"])
        MaskedLmExample = collections.namedtuple(
            "MaskedLmExample", ["token_ids", "label_ids", "positions", "weights"])
        rng = random.Random()
        candidate_indexes = []
        tokens = [self.tokenizer.CLS_TOKEN]
        tokens.extend(self.tokenizer.tokenize(text))
        tokens.append(self.tokenizer.SEP_TOKEN)
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length-1]
            tokens.append(self.tokenizer.SEP_TOKEN)
        for (i, token) in enumerate(tokens):
            if token == self.tokenizer.CLS_TOKEN or token == self.tokenizer.SEP_TOKEN:
                continue
            if len(candidate_indexes) >= 1 and token.startswith("##"):
                candidate_indexes[-1].append(i)
            else:
                candidate_indexes.append([i])
        rng.shuffle(candidate_indexes)
        output_tokens = list(tokens)
        num_to_predict = min(
            self.max_predictions_per_seq, max(1, int(
                round(len(tokens) * self.masked_lm_prob))))
        masked_lms = []
        covered_indexes = set()
        for index_set in candidate_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                # 100% of the time, replace with [MASK]
                if rng.random() < 0.8:
                    masked_token = self.tokenizer.MASK_TOKEN
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with another token
                    else:
                        masked_token = self.tokenizer.id2token(
                            rng.randint(0, self.tokenizer.vocab_size-1))
                output_tokens[index] = masked_token
                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_positions = []
        masked_lm_labels = []
        masked_lm_weights = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)
            masked_lm_weights.append(1)
        while len(masked_lm_positions) < self.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_labels.append(self.tokenizer.PAD_TOKEN)
            masked_lm_weights.append(0)
        while len(output_tokens) < self.seq_length:
            output_tokens.append(self.tokenizer.PAD_TOKEN)
        if len(output_tokens) > self.seq_length:
            output_tokens = output_tokens[:self.seq_length-1]
            output_tokens.append(self.tokenizer.SEP_TOKEN)
        assert len(output_tokens) == self.seq_length
        example = MaskedLmExample(
            token_ids=self.tokenizer.tokens2ids(output_tokens),
            label_ids=self.tokenizer.tokens2ids(masked_lm_labels),
            positions=masked_lm_positions, weights=masked_lm_weights)
        return example

    def create_batch_example(self, texts):
        token_ids = []
        label_ids = []
        positions = []
        weights = []
        for text in texts:
            example = self.create_example(text)
            token_ids.append(example.token_ids)
            label_ids.append(example.label_ids)
            positions.append(example.positions)
            weights.append(example.weights)
        MaskedLmExamples = collections.namedtuple(
            "MaskedLmExamples", ["token_ids", "label_ids", "positions", "weights"])
        return MaskedLmExamples(token_ids=token_ids, label_ids=label_ids, positions=positions, weights=weights)
