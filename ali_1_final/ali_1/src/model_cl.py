# 导入库
from init import *

#----------------------------------------
# 池化
class Pooler(nn.Module):

    # 初始化
    def __init__(self):
        super().__init__()
        return

    #--------------------
    # 前向
    def forward(self, attention_mask, outputs):

        last_hidden = outputs.last_hidden_state
        # attention_mask: shape (3 * batch_size, max_len)
        # last_hidden: shape (3 * batch_size, max_len, 768)

        last_hidden_masked = last_hidden * (attention_mask.unsqueeze(-1))
        last_hidden_masked_sum = last_hidden_masked.sum(1)
        attention_mask_sum = attention_mask.sum(-1).unsqueeze(-1)
        ans = last_hidden_masked_sum / attention_mask_sum
        # last_hidden_masked: shape (3 * batch_size, max_len, 768) | 若该词遮挡，则整个 embed 的元素都为 0
        # last_hidden_masked_sum: shape (3 * batch_size, 768) | 句子所有未被遮挡的词语的 embed 之和
        # attention_mask_sum: shape (3 * batch_size, 1) | 句子未被遮挡的词语的总个数
        # ans: shape (3 * batch_size, 768) | 句子的 embed

        return ans

#----------------------------------------
# 全连接层
class MLPLayer(nn.Module):

    # 初始化
    def __init__(self, config, pooler_num):
        super().__init__()

        hidden_size = config.hidden_size

        #----------
        self.dense = nn.Linear(hidden_size, pooler_num)
        self.tanh = nn.Tanh()

        return

    #--------------------
    # 前向
    def forward(self, features):

        dense = self.dense
        tanh = self.tanh

        # ----------
        x = dense(features)
        x = tanh(x)

        return x

#----------------------------------------
# 相似度
class Similarity(nn.Module):

    # 初始化
    def __init__(self, temp):
        super().__init__()

        self.cos = nn.CosineSimilarity(dim=-1)
        self.temp = temp

        return

    #--------------------
    # 前向
    def forward(self, x, y):

        cos = self.cos
        temp = self.temp

        #----------
        similiarity = cos(x, y)
        similiarity /= temp

        return similiarity

#----------------------------------------
# 对比学习初始化
def cl_init(cls, config, pooler_type, temp, pooler_num):

    cls.pooler_type = pooler_type
    cls.pooler = Pooler()
    if pooler_type == 'cls':
        cls.mlp = MLPLayer(config, pooler_num)
    # pooler_type: cls
    # pooler: avg
    cls.sim = Similarity(temp=temp)
    cls.init_weights()

    return

#----------------------------------------
# 对比学习句子前向
def sentemb_forward(cls, encoder, input_ids, token_type_ids, attention_mask):

    device = cls.device
    # input_ids, token_type_ids, attention_mask: shape (batch_size, max_len)

    #----------
    # 编码器所需参数
    kwargs_encoder = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'output_hidden_states': True,
        'return_dict': True
    }
    outputs = encoder(**kwargs_encoder)
    
    #----------
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = cls.mlp(pooler_output)
    # pooler_output: shape (batch_size, 128)

    #----------
    # 输出所需参数
    kwargs_ans = {
        'pooler_output': pooler_output,
        'hidden_states': outputs.hidden_states,
        'last_hidden_state': outputs.last_hidden_state
    }
    ans = BaseModelOutputWithPoolingAndCrossAttentions(**kwargs_ans)

    return ans

#----------------------------------------
# 对比学习前向
def cl_forward(cls, encoder, input_ids, token_type_ids, attention_mask):

    device = cls.device
    batch_size, num_sent, max_len = input_ids.shape
    # input_ids, token_type_ids, attention_mask: shape (batch_size, 3, max_len)

    input_ids = input_ids.reshape(-1, max_len)
    token_type_ids = token_type_ids.reshape(-1, max_len)
    attention_mask = attention_mask.reshape(-1, max_len)

    #----------
    # 编码器所需参数
    kwargs_encoder = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'output_hidden_states': True,
        'return_dict': True
    }
    outputs = encoder(**kwargs_encoder)

    #----------
    pooler_output = cls.pooler(attention_mask, outputs)
    hidden_size = pooler_output.shape[-1]
    # pooler_output: shape (3 * batch_size, 768) | 句子的 embed
    pooler_output = pooler_output.reshape(batch_size, num_sent, hidden_size)
    # pooler_output: shape (batch_size, 3, 768)
    pooler_output = cls.mlp(pooler_output)
    # pooler_output: shape (batch_size, 3, 128)

    sents0 = pooler_output[:, 0]
    sents1 = pooler_output[:, 1]
    sents2 = pooler_output[:, 2]
    # sents0: shape (batch_size, 128)

    #----------
    # 聚集多卡的随机负样本，每一份显存连续
    def gatherSents(sents):

        if dist.is_initialized() and cls.training:
            lst_sents = [torch.zeros_like(sents) for i_cuda in range(dist.get_world_size())]
            dist.all_gather(lst_sents, sents.contiguous())
            # 保留梯度
            lst_sents[dist.get_rank()] = sents
            sents = torch.cat(lst_sents, dim=0)

        return sents
    
    sents0 = gatherSents(sents0)
    sents1 = gatherSents(sents1)
    sents2 = gatherSents(sents2)
    # sents0: shape (N * batch_size, 128)

    #----------
    # 归一化
    sents0 = F.normalize(sents0)
    sents1 = F.normalize(sents1)
    sents2 = F.normalize(sents2)

    #----------
    # 相似度
    sim_pos = cls.sim(sents0.unsqueeze(1), sents1.unsqueeze(0))
    sim_neg = cls.sim(sents0.unsqueeze(1), sents2.unsqueeze(0))
    sim = torch.cat([sim_pos, sim_neg], dim=1)
    # sim_pos: shape (N * batch_size, N * batch_size)
    # sim: shape (N * batch_size, 2 * N * batch_size)

    #----------
    # 损失
    num_q = sim.shape[0]
    labels = torch.arange(num_q).long().to(device)
    # labels: shape (N * batch_size,) | 该 q 对应的 d+ 的相似度在所有 d 中最高

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(sim, labels)

    #----------
    # 输出所需参数
    kwargs_ans = {
        'loss': loss,
        'logits': sim,
        'hidden_states': outputs.hidden_states,
        'attentions': outputs.attentions,
        'z1': sents0.detach(),
        'z2': sents1.detach()
    }
    ans = SimCSEOutput(**kwargs_ans)

    return ans

#----------------------------------------
# 对比学习 Bert
class BertForCL(BertPreTrainedModel):

    # 初始化
    def __init__(self, config, pooler_type='cls', temp=0.05, pooler_num=128):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        _keys_to_ignore_on_load_missing = [r'position_ids']
        cl_init(self, config, pooler_type, temp, pooler_num)

        return

    #--------------------
    # 前向
    def forward(self, input_ids, attention_mask, token_type_ids, sent_emb=False):

        bert = self.bert

        #----------
        kwargs_cl_forward = {
            'cls': self,
            'encoder': bert,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        
        if sent_emb == False:
            return cl_forward(**kwargs_cl_forward)
        else:
            return sentemb_forward(**kwargs_cl_forward)

#----------------------------------------
@dataclass
class SimCSEOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    z1: torch.FloatTensor = None
    z2: torch.FloatTensor = None

#----------------------------------------
# 对抗训练
class FGM:

    # 初始化
    def __init__(self, model):

        self.model = model
        self.backup = {}

        return

    #--------------------
    # 攻击
    def attack(self, emb_name='word_embeddings', epsilon=1):

        model = self.model

        #----------
        for name, param in model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

        return

    #--------------------
    # 恢复
    def restore(self, emb_name='word_embeddings'):

        model = self.model

        #----------
        for name, param in model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}

        return
