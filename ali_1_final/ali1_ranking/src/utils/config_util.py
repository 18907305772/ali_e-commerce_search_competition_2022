import json


class BasedConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"No attribute named `{key}`")
            self._set_attribute(key, value)

    def _set_attribute(self, name, value):
        try:
            if getattr(self, name, None) is None:
                setattr(self, name, value)
        except AttributeError as err:
            print(f"Can't set `{name}` with value `{value}` for {self.__name__}")
            raise err

    def from_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as reader:
            config_dict = json.load(reader)
        for key, value in config_dict.items():
            if not hasattr(self, key):
                continue
            self._set_attribute(key, value)
        return self

    def __str__(self):
        return f"({', '.join([f'{str(name)}={str(value)}' for name, value in self.__dict__.items()])})"


class TransformerConfig(BasedConfig):
    def __init__(self, **kwargs):
        self.vocab_size = None
        self.seq_length = None
        self.hidden_size = None
        self.num_hidden_layers = None
        self.num_attention_heads = None
        self.hidden_act = None
        self.intermediate_size = None
        self.hidden_dropout_prob = None
        self.attention_probs_dropout_prob = None
        self.max_position_embeddings = None
        self.type_vocab_size = None
        self.initializer_range = None
        self.layer_norm_eps = None
        self.pad_token_id = None
        super().__init__(**kwargs)


class BertConfig(TransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ErnieConfig(TransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
