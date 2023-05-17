# 导入库
from init import *

#----------------------------------------
# 数据处理
@dataclass
class PadDataCollator:

    # 初始化
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer
        self.special_keys = ['input_ids', 'token_type_ids', 'attention_mask']

        return

    #--------------------
    def __call__(self, features):

        special_keys = self.special_keys
        tokenizer = self.tokenizer

        # ----------
        batch_size = len(features)
        num_sent = len(features[0]['input_ids'])

        features_flat = []
        for feature in features:
            for i_sent in range(num_sent):
                features_flat.append({key: feature[key][i_sent] for key in feature.keys() if key in special_keys})

        batch = tokenizer.pad(features_flat, return_tensors='pt', pad_to_multiple_of=None)
        for key in batch.keys():
            if key in special_keys:
                batch[key] = batch[key].reshape(batch_size, num_sent, -1)
                # batch[key]: shape (batch_size, 3, max_len)

        return batch

