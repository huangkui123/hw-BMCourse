import sys, json, copy, six, logging

from model import Bert
from data import LoadDataset
from log_helper import logger_init
from transformers import BertTokenizer
import logging
import torch
import os
import time

class BertConfig(object):
    def __init__(self,
                 vocab_size=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 pad_token_id=0,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
      
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data/glue-sst2')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert-base-uncased")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'dev.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.out_path = os.path.join(self.project_dir, 'out.txt')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 30
        self.max_sen_len = None
        self.num_labels = 2
        self.epochs = 10 #################################################################################
        self.model_val_per_epoch = 2
        logger_init(log_file_name='single', log_level=logging.INFO, log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value

        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    model = Bert(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadDataset(vocab_path=config.vocab_path,
                              tokenizer=bert_tokenize,
                              batch_size=config.batch_size,
                              max_sen_len=config.max_sen_len,
                              split_sep=config.split_sep,
                              max_position_embeddings=config.max_position_embeddings,
                              pad_index=config.pad_token_id,
                              is_sample_shuffle=config.is_sample_shuffle)

    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path, config.val_file_path, config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 1000 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

        acc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
        logging.info(f"Accuracy on val {acc:.3f}")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)

        acc = evaluate(test_iter, model, config.device, data_loader.PAD_IDX)
        logging.info(f"Accuracy on test {acc:.3f}")



def inference(config):
    model = Bert(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测")
    model = model.to(config.device)
    data_loader = LoadDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path, config.val_file_path, config.test_file_path)

    acc = evaluate(test_iter, model, config.device, data_loader.PAD_IDX)
    logging.info(f"Accuracy on test {acc:.3f}")

def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


if __name__ == '__main__':
    model_config = ModelConfig()
    train(model_config)
    inference(model_config)
