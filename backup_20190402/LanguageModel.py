import json
import pickle
import random

import torch
from torch import nn, optim
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import time

import copy

from Vocab import Vocab




class LanguageModel(nn.Module):
    def __init__(self, use_cuda, hidden_dim, input_dim, vocab):#, pre_train_weight, is_fix_word_vector = 1):
        super(LanguageModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.vocab = vocab
        
        self.lstm=torch.nn.LSTM(input_size=self.input_dim, 
                                hidden_size= self.hidden_dim, 
                                bidirectional=False,
                                batch_first=True
                               )
        
        #embedding
        self.embed=nn.Embedding(len(self.vocab.word2token), input_dim)
        #loading pre trained word embedding
        with open('data_set/pre_trained_token_embedding_300d.pk', 'rb') as f:
            pre_train_word_embedding = pickle.load(f)
            
        self.embed.weight.data.copy_(torch.FloatTensor(pre_train_word_embedding))
#         self.embed.weight.requires_grad = False


        self.weight = [1]*len(self.vocab.word2token)
        self.weight[self.vocab.word2token['<padding>']]=0
        self.cost_func = nn.CrossEntropyLoss(weight=torch.Tensor(self.weight), reduce=True)
        self.fcnn=nn.Linear(in_features = self.hidden_dim, out_features = len(self.vocab.word2token))
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def order(self, inputs, inputs_len):    #inputs: tensor, inputs_len: 1D tensor
        inputs_len, sort_ids = torch.sort(inputs_len, dim=0, descending=True)
        
        if self.use_cuda:
            inputs = inputs.index_select(0, Variable(sort_ids).cuda())
        else:
            inputs = inputs.index_select(0, Variable(sort_ids))
        
        _, true_order_ids = torch.sort(sort_ids, dim=0, descending=False)
        
        return inputs, inputs_len, true_order_ids
    #
    def forward(self, inputs, inputs_len):
        inputs = Variable(inputs)
        if self.use_cuda:
            inputs=inputs.cuda()
            
        inputs, sort_len, true_order_ids = self.order(inputs, inputs_len)

        in_vecs=self.embed(inputs)

        packed = rnn_utils.pack_padded_sequence(input=in_vecs, lengths=list(sort_len), batch_first =True)
        
        outputs, (hn,cn) = self.lstm(packed)
        outputs, sent_lens = rnn_utils.pad_packed_sequence(outputs)
        
        #print('outpurs size, hn size and cn size: ', outputs.size(), hn.size(), cn.size())
        outputs = outputs.transpose(0,1)  #transpose is necessary
        #print('outpurs size, hn size and cn size: ', outputs.size(), hn.size(), cn.size())
        
        #warnning: outputs, hn and cn have been sorted by sentences length so the order is wrong, now to sort them
        if self.use_cuda:
            outputs = outputs.index_select(0, Variable(true_order_ids).cuda())
        else:
            outputs = outputs.index_select(0, Variable(true_order_ids))
        
#         hn = torch.cat((hn[0], hn[1]), dim=1)
#         cn = torch.cat((cn[0], cn[1]), dim=1)
#         #print('hn size and cn size: ', hn.size(), cn.size())
        
#         if self.use_cuda:
#             hn = hn.index_select(0, Variable(true_order_ids).cuda())
#             cn = cn.index_select(0, Variable(true_order_ids).cuda())
#         else:
#             hn = hn.index_select(0, Variable(true_order_ids))
#             cn = cn.index_select(0, Variable(true_order_ids))
        logits = self.fcnn(outputs)
        return logits
    
    def get_loss(self, logits, labels):
        labels = self._tocuda(Variable(labels))
        sent_len = logits.size(dim=1)
        labels = labels[:, :sent_len]
        labels = labels.contiguous().view(-1)
        logits = logits.view(-1, len(self.vocab.word2token))
#         print('logits size: ', logits.size())
        
        return self.cost_func(logits, labels)
    
    def get_sentences_ppl(self, inputs, inputs_len, labels):
        
        vocab_size=len(self.vocab.word2token)
        batch_size=inputs.size(0)
        
        logits = self.forward(inputs, inputs_len)
        
        labels = self._tocuda(Variable(labels))
        sent_len = logits.size(dim=1)
        labels = labels[:, :sent_len]
        labels = labels.contiguous()
        logits = logits.view(-1, vocab_size)
        log_probs= self.log_softmax(logits).view(-1)
        
        pos_bias=torch.LongTensor([i*vocab_size for i in range(sent_len)]).view(1,-1)
        pos_bias = pos_bias.expand(batch_size, pos_bias.size(1))
#         print(pos_bias.size())

        batch_bias = torch.LongTensor([i*vocab_size*sent_len for i in range(batch_size)]).view(-1,1)
        batch_bias = batch_bias.expand(batch_bias.size(0), sent_len)
#         print(batch_bias.size())

        pos_bias = self._tocuda(Variable(pos_bias, requires_grad=0))
        batch_bias = self._tocuda(Variable(batch_bias, requires_grad=0))
        
        indices=labels+pos_bias+batch_bias
        indices = indices.view(-1)

        results=log_probs[indices].view(batch_size, sent_len)
        
        sents_ppl=[]
        for idx, result in enumerate(results):
            mean_log_prob = torch.mean(result[:inputs_len[idx]])
            sents_ppl.append(mean_log_prob)

        sents_ppl = torch.cat(sents_ppl, dim=0)
        sents_ppl = torch.exp(-sents_ppl)
        
        if self.use_cuda:
            return sents_ppl.cpu().data.tolist()
        else:
            return sents_ppl.data.tolist()
        
    def _tocuda(self, var):
        if self.use_cuda:
            return var.cuda()
        else:
            return var