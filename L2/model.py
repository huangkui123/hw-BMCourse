import torch
from torch.nn.init import normal_
import torch.nn as nn
import os
import logging
from copy import deepcopy

def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, hidden_states, hidden_states, attn_mask=None, key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, intermediate_size]
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.bert_attention(hidden_states, attention_mask)
        # [src_len, batch_size, hidden_size]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, hidden_size]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output, attention_mask)
            #  [src_len, batch_size, hidden_size]
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        pooled_output = self.dense(token_tensor)  # [batch_size, hidden_size]
        pooled_output = self.activation(pooled_output)
        return pooled_output  # [batch_size, hidden_size]

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def format_paras_for_torch(loaded_paras_names, loaded_paras):
    qkv_weight_names = ['query.weight', 'key.weight', 'value.weight']
    qkv_bias_names = ['query.bias', 'key.bias', 'value.bias']
    qkv_weight, qkv_bias = [], []
    torch_paras = []
    for i in range(len(loaded_paras_names)):
        para_name_in_pretrained = loaded_paras_names[i]
        para_name = ".".join(para_name_in_pretrained.split('.')[-2:])
        if para_name in qkv_weight_names:
            qkv_weight.append(loaded_paras[para_name_in_pretrained])
        elif para_name in qkv_bias_names:
            qkv_bias.append(loaded_paras[para_name_in_pretrained])
        else:
            torch_paras.append(loaded_paras[para_name_in_pretrained])
        if len(qkv_weight) == 3:
            torch_paras.append(torch.cat(qkv_weight, dim=0))
            qkv_weight = []
        if len(qkv_bias) == 3:
            torch_paras.append(torch.cat(qkv_bias, dim=0))
            qkv_bias = []
    return torch_paras

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config
        self._reset_parameters()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        embedding_output = self.bert_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        # embedding_output: [src_len, batch_size, hidden_size]
        all_encoder_outputs = self.bert_encoder(embedding_output, attention_mask=attention_mask)
        # all_encoder_outputs 为一个包含有num_hidden_layers个层的输出
        sequence_output = all_encoder_outputs[-1]  # 取最后一层
        # sequence_output: [src_len, batch_size, hidden_size]
        pooled_output = self.bert_pooler(sequence_output)
        # pooled_output: [batch_size, hidden_size]
        return pooled_output, all_encoder_outputs

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=self.config.initializer_range)

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)
        pretrained_model_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<路径：{pretrained_model_path} 中的模型不存在")
        loaded_paras = torch.load(pretrained_model_path)
        state_dict = deepcopy(model.state_dict())
        loaded_paras_names = list(loaded_paras.keys())[:-8]
        model_paras_names = list(state_dict.keys())[1:]

        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            torch_paras = format_paras_for_torch(loaded_paras_names, loaded_paras)
            for i in range(len(model_paras_names)):
                logging.debug(f"## 成功赋值参数:{model_paras_names[i]},形状为: {torch_paras[i].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = torch_paras[i]

        model.load_state_dict(state_dict)
        return model

class Bert(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(Bert, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None):
        pooled_output, _ = self.bert(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     token_type_ids=token_type_ids, 
                                     position_ids=position_ids)  
        # [batch_size,hidden_size] 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
