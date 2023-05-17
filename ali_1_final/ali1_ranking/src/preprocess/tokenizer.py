import collections
import unicodedata

__all__ = ['Tokenizer']


def padding(ids, fixed_length):
    while len(ids) < fixed_length:
        ids.append(0)
    return ids


def truncate(ids, fixed_length):
    if len(ids) > fixed_length:
        ids = ids[:fixed_length]
    return ids


class Tokenizer:
    UNK_TOKEN = '[UNK]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    PAD_TOKEN = '[PAD]'

    def __init__(self, vocab_file, uncased=True):
        """
        :param vocab_file: vocabulary file
        """
        self._do_lower_case = uncased
        if self._do_lower_case:
            Tokenizer.UNK_TOKEN = Tokenizer.UNK_TOKEN.lower()
            Tokenizer.CLS_TOKEN = Tokenizer.CLS_TOKEN.lower()
            Tokenizer.SEP_TOKEN = Tokenizer.SEP_TOKEN.lower()
            Tokenizer.MASK_TOKEN = Tokenizer.MASK_TOKEN.lower()
            Tokenizer.PAD_TOKEN = Tokenizer.PAD_TOKEN.lower()
        self._special_tokens = [
            Tokenizer.UNK_TOKEN,
            Tokenizer.CLS_TOKEN,
            Tokenizer.SEP_TOKEN,
            Tokenizer.MASK_TOKEN,
            Tokenizer.PAD_TOKEN]
        self.vocab_file = vocab_file
        self.vocab_id, self.id_vocab, self.vocab_size, self.unk_index \
            = self._load_vocab()

    def _load_vocab(self):
        vocab_id = collections.OrderedDict()
        id_vocab = collections.OrderedDict()
        vocab_size = 0
        unk_index = -1
        with open(self.vocab_file, "r", encoding="utf-8") as reader:
            for index, token in enumerate(reader.readlines()):
                token = token.strip()
                token = token.lower() if self._do_lower_case else token
                if token == Tokenizer.UNK_TOKEN:
                    unk_index = index
                vocab_id[token] = index
                id_vocab[index] = token
                vocab_size += 1
        assert unk_index != -1
        return vocab_id, id_vocab, vocab_size, unk_index

    def tokenize(self, text):
        """ Tokenize a piece of text. """
        if self._do_lower_case:
            text = text.lower()
        text = clean_text(text)
        tokens = list(text)

        # Tokenize Chinese chars
        tokens = self._chinese_tokenize(tokens)

        # Tokenize by whitespace
        tokens = self._whitespace_tokenize(tokens)

        # Tokenize by punctuation
        tokens = self._punctuation_tokenize(tokens, ['[', ']'])

        # Tokenize by special tokens
        tokens = self._special_tokenize(tokens)

        # word piece tokenize
        tokens = self._word_piece_tokenize(tokens)
        return tokens

    def texts2ids(self, texts: list, seq_length: int):
        token_ids_list = []
        for text in texts:
            token_ids = self.text2ids(text, seq_length)
            token_ids_list.append(token_ids)
        return token_ids_list

    def text2ids(self, text: str, seq_length: int):
        token_ids = [self.cls_id()]
        token_ids.extend(self.tokens2ids(self.tokenize(text)))
        token_ids.append(self.sep_id())
        token_ids = padding(token_ids, seq_length)
        token_ids = truncate(token_ids, seq_length)
        assert len(token_ids) == seq_length
        return token_ids

    def ids2token_type_ids(self, token_ids, max_types=2):
        token_type_ids = []
        token_type_id = 0
        for token_id in token_ids:
            token_type_ids.append(token_type_id)
            if token_id == self.sep_id() and token_type_id < max_types - 1:
                token_type_id += 1
        return token_type_ids

    # Tokenize Chinese chars
    @staticmethod
    def _chinese_tokenize(tokens: list):
        output = []
        start_new_word = True
        word = ''
        for char in tokens:
            if is_chinese_char(char):
                # TODO: 可在此处添加词切分
                if len(word) != 0:
                    output.append(word)
                output.append(char)
                start_new_word = True
                word = ''
            else:
                if start_new_word:
                    start_new_word = False
                word += char
        output.append(word)
        return output

    # Tokenize by whitespace
    @staticmethod
    def _whitespace_tokenize(tokens: list):
        return " ".join(tokens).split()

    # Tokenize by punctuation
    @staticmethod
    def _punctuation_tokenize(tokens: list, ignore_chars=None):
        if ignore_chars is None:
            ignore_chars = []
        output = []
        for _token in tokens:
            start_new_word = True
            word = ''
            for char in _token:
                if char not in ignore_chars and is_punctuation(char):
                    if len(word) != 0:
                        output.append(word)
                    output.append(char)
                    start_new_word = True
                    word = ''
                else:
                    if start_new_word:
                        start_new_word = False
                    word += char
            if len(word) != 0:
                output.append(word)
        return output

    # Word piece tokenize
    def _word_piece_tokenize(self, tokens: list):
        # Check for special tokens, such as [MASK]
        is_split_able = []
        for token in tokens:
            if len(token) == 1:
                is_split_able.append(0)
            elif self._is_special_token(token):
                is_split_able.append(0)
            else:
                is_split_able.append(1)

        output = []
        for i, token in enumerate(tokens):
            if not is_split_able[i]:
                output.append(token)
                continue
            for t in self._punctuation_tokenize([token]):
                chars = list(t)
                is_bad = False
                start = 0
                sub_tokens = []
                while start < len(chars):
                    end = len(chars)
                    cur_substr = None
                    while start < end:
                        substr = "".join(chars[start:end])
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab_id:
                            cur_substr = substr
                            break
                        end -= 1
                    if cur_substr is None:
                        is_bad = True
                        break
                    sub_tokens.append(cur_substr)
                    start = end

                if is_bad:
                    output.append(Tokenizer.UNK_TOKEN)
                else:
                    output.extend(sub_tokens)
        return output

    # Find special token
    def _special_tokenize(self, tokens: list):
        output = []
        for token in tokens:
            _token = token
            for special_token in self._special_tokens:
                if special_token in token:
                    _token = _token.replace(special_token, ' ' + special_token + ' ')
            _token = _token.split()
            output.extend(_token)
        return output

    def _is_special_token(self, token):
        return token in self._special_tokens

    def cls_id(self):
        """ Get the index of Begin-Token """
        return self.token2id(Tokenizer.CLS_TOKEN)

    def sep_id(self):
        """ Get the index of End-Token """
        return self.token2id(Tokenizer.SEP_TOKEN)

    def token2id(self, token: str):
        token = token.lower() if self._do_lower_case else token
        return self.vocab_id.get(
            token, self.unk_index)

    def id2token(self, idx: int):
        return self.id_vocab.get(idx, Tokenizer.UNK_TOKEN)

    def tokens2ids(self, tokens: list):
        ids = []
        for token in tokens:
            ids.append(self.token2id(token))
        return ids

    def ids2tokens(self, ids: list):
        tokens = []
        for idx in ids:
            tokens.append(self.id2token(idx))
        return tokens


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(char):
    """Checks whether a char is a Chinese character."""
    cp = ord(char)
    if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F)):
        return True
    return False


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def clean_text(text):
    text = text.strip()
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)
