import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers.pre_tokenizers import Whitespace
from math import floor
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification
)
import wandb
from tqdm import tqdm

from .ner_dataset import NerCorpusDataset
from .utils import *


class HealthNERModel(object):
    def __init__(self, label_dict: dict,
                 tokenizers_parallelism=True,
                 learning_rate=1e-05,
                 train_per=0.9,
                 max_len=128,
                 num_workers=2,
                 is_split=False,
                 epochs=1,
                 train_batch_size=2,
                 test_batch_size=1,
                 max_grad_norm=10,
                 optimizer_name='adam'
                 ):
        '''
        params
            label_dict: dict

            tokenizers_parallelism: bool
            learning_rate: float
            train_per: float
            max_len: int
            num_workers: int
            is_split: bool
            epochs: int
            train_batch_size: int
            test_batch_size: int
            max_grad_norm: int
            optimizer_name: str; assert in ['adam', 'sgd']
        '''
        self.label_dict = label_dict
        self.tokenizers_parallelism = tokenizers_parallelism
        self.learning_rate = learning_rate
        self.train_per = train_per
        self.max_len = max_len
        self.num_workers = num_workers
        self.is_split = is_split
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_grad_norm = max_grad_norm
        self.optimizer_name = optimizer_name

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-chinese', tokenizers_parallelism=True)
        self.model = BertForTokenClassification.from_pretrained(
            'ckiplab/bert-base-chinese', num_labels=len(label_dict))
        self.model.to(self.device)

        self._build_dataset()
        self._build_optimizer(optimizer_name)

    def main(self):
        '''
        main function for train and test
        '''
        wandb.watch(self.model)

        for e in range(1, self.epochs + 1):
            print('epochs:\t', e)
            self._train(e)
            self._test(e)

    def _train(self, epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps, notnan_steps = 0, 0, 0
        self.model.train()

        train_bar = tqdm(self.train_loader)
        for idx, batch in enumerate(train_bar):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            labels = batch['labels'].to(self.device, dtype=torch.long)
            outputs = self.model(
                input_ids=ids, attention_mask=mask, labels=labels)

            loss = outputs.loss
            tr_logits = outputs.logits
            tr_loss += loss.item()
            nb_tr_steps += 1
            wandb.log({"train_loss": loss})

            tmp_tr_accuracy = cal_acc(labels, tr_logits, self.model.num_labels)

            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.max_grad_norm
            )

            if tmp_tr_accuracy is not None:
                wandb.log({'train_acc': tmp_tr_accuracy})
                tr_accuracy += tmp_tr_accuracy
                notnan_steps += 1

            if nb_tr_steps % 5000 == 0 or nb_tr_steps == len(self.train_loader):
                if nb_tr_steps == len(self.train_loader):
                    save_model(self.model, './', 'model_ner_' + self.optimizer_name +
                               '_' + str(self.learning_rate) + '_' + str(epoch) + '.pt')
                else:
                    save_model(self.model.state_dict(), './',
                               'checkpoint_ner_' + str(epoch) + '.pt')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_bar.set_postfix({'loss': '{0:.4f}'.format(loss.item()),
                                   'loss_a': '{0:.4f}'.format(tr_loss/nb_tr_steps),
                                   'accuracy': ('{0:.4f}'.format(tmp_tr_accuracy)) if tmp_tr_accuracy is not None else '------',
                                   'accuracy_a': ('{0:.4f}'.format(tr_accuracy / notnan_steps)) if notnan_steps is not 0 else '------'})

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / notnan_steps

        print(f"Training acc:  {tr_accuracy}")
        print(f"Training loss: {epoch_loss}")

    def _test(self, epoch):
        ts_loss, ts_accuracy = 0, 0
        nb_ts_examples, nb_ts_steps, notnan_steps = 0, 0, 0
        self.model.eval()

        test_bar = tqdm(self.test_loader)
        for idx, batch in enumerate(test_bar):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            labels = batch['labels'].to(self.device, dtype=torch.long)
            outputs = self.model(
                input_ids=ids, attention_mask=mask, labels=labels)

            loss = outputs.loss
            ts_logits = outputs.logits
            ts_loss += loss.item()
            nb_ts_steps += 1
            wandb.log({"test_loss": loss})

            tmp_ts_accuracy = cal_acc(labels, ts_logits, self.model.num_labels)

            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.max_grad_norm
            )

            if tmp_ts_accuracy is not None:
                wandb.log({'test_acc': tmp_ts_accuracy})
                ts_accuracy += tmp_ts_accuracy
                notnan_steps += 1

            test_bar.set_postfix({'loss': '{0:.4f}'.format(loss.item()),
                                  'loss_a': '{0:.4f}'.format(ts_loss/nb_ts_steps),
                                  'accuracy': ('{0:.4f}'.format(tmp_ts_accuracy)) if tmp_ts_accuracy is not None else '------',
                                  'accuracy_a': ('{0:.4f}'.format(ts_accuracy / notnan_steps)) if notnan_steps is not 0 else '------'})

        epoch_loss = ts_loss / nb_ts_steps
        ts_accuracy = ts_accuracy / notnan_steps

        print(f"Testing acc:  {ts_accuracy}")
        print(f"Testing loss: {epoch_loss}")

    def _build_dataset(self):
        train_set = NerCorpusDataset(
            self.tokenizer, '../input/chinese-healthner-corpus/train/train.json', self.label_dict, self.max_len)
        test_set = NerCorpusDataset(
            self.tokenizer, '../input/chinese-healthner-corpus/test.json', self.label_dict, self.max_len)

        if self.is_split:
            train_num = floor(train_set.__len__() * self.train_per)
            valid_num = train_set.__len__() - train_num
            train_set, valid_set = torch.utils.data.random_split(
                train_set, [train_num, valid_num])
            self.valid_loader = DataLoader(
                valid_set, batch_size=self.test_batch_size, shuffle=True, num_workers=self.num_workers)

        print(train_set.__len__(), valid_set.__len__()
              if self.is_split else '', test_set.__len__())
        self.train_loader = DataLoader(
            train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(
            test_set, batch_size=self.test_batch_size, shuffle=True, num_workers=self.num_workers)

    def _build_optimizer(self, optimizer):
        assert optimizer in ['sgd', 'adam']
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.learning_rate, momentum=0.9)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.learning_rate)
