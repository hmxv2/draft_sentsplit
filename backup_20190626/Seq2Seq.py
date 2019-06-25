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
from LanguageModel import LanguageModel

import torch

print('import over')

copy_thres=1


with open('data_set/vocab.pk', 'rb') as f:
    vocab=pickle.load(f)
    
    
    
class Encoder(nn.Module):
    def __init__(self, use_cuda, hidden_dim, input_dim, vocab):#, pre_train_weight, is_fix_word_vector = 1):
        super(Encoder, self).__init__()
        
        self.use_cuda = use_cuda
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.vocab = vocab
        
        self.lstm=torch.nn.LSTM(input_size=self.input_dim, 
                                hidden_size= self.hidden_dim, 
                                bidirectional=True,
                                batch_first=True
                               )
        
        #embedding
        self.embed=nn.Embedding(len(self.vocab.word2token), input_dim)
        #loading pre trained word embedding
        with open('data_set/pre_trained_token_embedding.pk', 'rb') as f:
            pre_train_word_embedding = pickle.load(f)
            
        self.embed.weight.data.copy_(torch.FloatTensor(pre_train_word_embedding))
#         self.embed.weight.requires_grad = False
        
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
        
        hn = torch.cat((hn[0], hn[1]), dim=1)
        cn = torch.cat((cn[0], cn[1]), dim=1)
        #print('hn size and cn size: ', hn.size(), cn.size())
        
        if self.use_cuda:
            hn = hn.index_select(0, Variable(true_order_ids).cuda())
            cn = cn.index_select(0, Variable(true_order_ids).cuda())
        else:
            hn = hn.index_select(0, Variable(true_order_ids))
            cn = cn.index_select(0, Variable(true_order_ids))
            
        return outputs, (hn,cn)
        
def _inflate(tensor, times, dim):
    """
    Examples::
        >> a = torch.LongTensor([[1, 2], [3, 4]])
        >> a
        1   2
        3   4
        [torch.LongTensor of size 2x2]
        >> b = ._inflate(a, 2, dim=1)
        >> b
        1   2   1   2
        3   4   3   4
        [torch.LongTensor of size 2x4]
    """
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)

class Decoder(nn.Module):
    def __init__(self, use_cuda, encoder, hidden_dim, max_length=25):
        super(Decoder, self).__init__()
        
        self.use_cuda = use_cuda
        self.hidden_dim=hidden_dim
        self.input_dim = encoder.input_dim
        self.max_length = max_length
        self.vocab = encoder.vocab
        self.weight = [1]*len(self.vocab.word2token)
        self.weight[self.vocab.word2token['<padding>']]=0
        #self.weight[self.vocab.word2token['<eos>']]=1.01
        #self.weight[self.vocab.word2token['<split>']]=1.01
        
        self.hidden_size = self.hidden_dim
        self.V = len(self.vocab.word2token)
        self.SOS = self.vocab.word2token['<sos>']
        self.EOS = self.vocab.word2token['<eos>']
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.lstmcell = torch.nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_dim*2, bias=True)
        
        #embedding
        self.embed=encoder.embed# reference share
        #fcnn: projection for crossentroy loss
        self.fcnn = nn.Linear(in_features = self.hidden_dim*2+hidden_dim*2, out_features = len(self.vocab.word2token))
        
        self.softmax = nn.Softmax(dim=1)
        self.cost_func = nn.CrossEntropyLoss(weight=torch.Tensor(self.weight), reduce=False)
        self.nll_loss = nn.NLLLoss(weight=torch.Tensor(self.weight), reduce=False)

        print('init lookup embedding matrix size: ', self.embed.weight.data.size())
        
        #copy
        out_features_dim=self.hidden_dim
        self.attent_wh = nn.Linear(in_features = self.hidden_dim*2, out_features = out_features_dim, bias = 0)
        self.attent_ws = nn.Linear(in_features = self.hidden_dim*2, out_features = out_features_dim, bias = 1)
        self.tanh = nn.Tanh()
        self.attent_vt = nn.Linear(in_features = out_features_dim, out_features = 1, bias=0)
        
        self.prob_wh = nn.Linear(in_features = self.hidden_dim*2, out_features = 1, bias=0)
        self.prob_ws = nn.Linear(in_features = self.hidden_dim*2, out_features = 1, bias=0)
        self.prob_wx = nn.Linear(in_features = self.input_dim, out_features = 1, bias=1)
        self.sigmoid = nn.Sigmoid()
    
    def get_context_vec(self, enc_outputs, this_timestep_input, dec_state):
        batch_size = enc_outputs.size(dim = 0)
        
        wh = self.attent_wh(enc_outputs)
        ws = self.attent_ws(dec_state).unsqueeze(dim=1)
#         print('wh, ws size: ', wh.size(), ws.size())
        ws = ws.expand(ws.size(0), wh.size(1), ws.size(2))
#         print('ws size: ', ws.size())
        weight = self.attent_vt(self.tanh(wh+ws))
#         print('weight size: ', weight.size())
        weight = self.softmax(weight.squeeze(dim=2))
#         print('weight size: ', weight.size())
        context_v = torch.bmm(weight.unsqueeze(dim=1), enc_outputs)
#         print('context_v size: ', context_v.size())
        context_v = context_v.squeeze(dim=1)
        return context_v, weight
    
    def copy_mechanism(self, enc_outputs, this_timestep_input, dec_state, inputs_one_hot, context_v, weight):
        batch_size = enc_outputs.size(dim = 0)
        
#         wh = self.attent_wh(enc_outputs)
#         ws = self.attent_ws(dec_state).unsqueeze(dim=1)
# #         print('wh, ws size: ', wh.size(), ws.size())
#         ws = ws.expand(ws.size(0), wh.size(1), ws.size(2))
# #         print('ws size: ', ws.size())
#         weight = self.attent_vt(self.tanh(wh+ws))
# #         print('weight size: ', weight.size())
#         weight = self.softmax(weight.squeeze(dim=2))
# #         print('weight size: ', weight.size())
#         context_v = torch.bmm(weight.unsqueeze(dim=1), enc_outputs)
# #         print('context_v size: ', context_v.size())
#         context_v = context_v.squeeze(dim=1)
        
        p_wh = self.prob_wh(context_v)
        p_ws = self.prob_ws(dec_state)
        p_wx = self.prob_wx(this_timestep_input)
        if_copy = self.sigmoid(p_wh+p_ws+p_wx)
#         if_copy = 0.3*if_copy
#         if_copy = self._tocuda(Variable(torch.ones(batch_size, 1), requires_grad=0))
#         print('if_copy size: ', if_copy.size())
        
        prob_copy = torch.bmm(inputs_one_hot, weight.unsqueeze(dim=2))
        prob_copy = prob_copy.squeeze(dim=2)
#         prob_copy = self._tocuda(Variable(torch.rand(batch_size, len(self.vocab.word2token)), requires_grad=0))
#         prob_copy = self.softmax(prob_copy)

#         print('prob_copy size: ', prob_copy.size())
#         print(torch.sum(prob_copy, dim=1))
#         print(torch.mean(if_copy))
        
#         if random.random()<0.005:
#             print('if_copy mean: ', torch.mean(if_copy))
#             _, max_ids = torch.max(prob_copy, dim=1)
#             print(self.vocab.token2word[max_ids.data[0]], self.vocab.token2word[max_ids.data[1]], self.vocab.token2word[max_ids.data[2]])
            
            
        return if_copy, prob_copy

    def forward(self, enc_outputs, sent_lens, h0_and_c0, labels, inputs, teaching_rate=0.6, is_train=1):
        labels = Variable(labels)
        if self.use_cuda:
            labels = labels.cuda()

        all_loss = 0
        predicts = []
        max_probs=[]
        batch_size = enc_outputs.size(dim = 0)
        final_hidden_states = h0_and_c0[0]
#         print('enc_outputs size:', enc_outputs.size())

        sents_len = enc_outputs.size(1)
        inputs = inputs[:,:sents_len].unsqueeze(dim=2)
        one_hot = torch.FloatTensor(batch_size, sents_len, len(self.vocab.word2token)).zero_()
        one_hot.scatter_(2, inputs, 1)
        one_hot = one_hot.transpose(1,2)
        one_hot = self._tocuda(Variable(one_hot, requires_grad = 0))
#         print('one_hot size: ', one_hot.size())
        
        for ii in range(self.max_length):
            if ii==0:
                zero_timestep_input = Variable(torch.LongTensor([self.vocab.word2token['<sos>']]*batch_size))
                if self.use_cuda:
                    zero_timestep_input = zero_timestep_input.cuda()
                    
                zero_timestep_input = self.embed(zero_timestep_input)#size: batch_size * self.input_dim
                
                last_timestep_hidden_state,cx = self.lstmcell(zero_timestep_input, h0_and_c0)
                #print('last_timestep_hidden_state: ', last_timestep_hidden_state.size(), cx.size())

                #get context vector
                context_vec, weight = self.get_context_vec(enc_outputs=enc_outputs, this_timestep_input=-1, 
                                                            dec_state = last_timestep_hidden_state)
                logits = self.fcnn(torch.cat([last_timestep_hidden_state, context_vec], dim=1))
                
                #copy or not
                copy_control=random.random()
                if copy_control<copy_thres:
                    if_copy, prob_copy = self.copy_mechanism(enc_outputs=enc_outputs, this_timestep_input=zero_timestep_input, 
                                                            dec_state = last_timestep_hidden_state, inputs_one_hot = one_hot, 
                                                            context_v=context_vec,
                                                            weight = weight)
                    score = (1-if_copy)*self.softmax(logits)+if_copy*prob_copy
                    score = torch.clamp(score, min=10**(-30), max=1)

                #for saving time: no training, no loss calculating
                if is_train:
                    if copy_control<copy_thres:
                        loss = self.nll_loss(torch.log(score), labels[:,0])
                    else:
                        loss = self.cost_func(logits, labels[:,0])
                    all_loss+=loss
                
                #get predicts
                if copy_control<copy_thres:
                    _, max_idxs = torch.max(score, dim=1)
                else:
                    _, max_idxs = torch.max(logits, dim=1)
                predicts.append(torch.unsqueeze(max_idxs, dim=0))
                
                
            else:
                if is_train:
                    rand = random.random()
                    if rand<teaching_rate:
                        this_timestep_input = self.embed(labels[:,ii-1])#label teaching, lookup embedding
                    else:
                        this_timestep_input = self.embed(max_idxs)#last_timestep output, and then look up word embedding
                else:
                    this_timestep_input = self.embed(max_idxs)#last_timestep output, and then look up word embedding
                
                last_timestep_hidden_state,cx = self.lstmcell(this_timestep_input, (last_timestep_hidden_state,cx))
                
                #get context vector
                context_vec, weight = self.get_context_vec(enc_outputs=enc_outputs, this_timestep_input=this_timestep_input, 
                                                            dec_state = last_timestep_hidden_state)
                logits = self.fcnn(torch.cat([last_timestep_hidden_state, context_vec], dim=1))
                
                #copy or not
                copy_control=random.random()
                if copy_control<copy_thres:
                    if_copy, prob_copy = self.copy_mechanism(enc_outputs=enc_outputs, this_timestep_input=this_timestep_input, 
                                                            dec_state = last_timestep_hidden_state, inputs_one_hot = one_hot, 
                                                             context_v=context_vec,
                                                            weight = weight)
                    score = (1-if_copy)*self.softmax(logits)+if_copy*prob_copy
                    score = torch.clamp(score, min=10**(-30), max=1)

                #for saving time: no training, no loss calculating
                if is_train:
                    if copy_control<copy_thres:
                        loss = self.nll_loss(torch.log(score), labels[:,ii])
                    else:
                        loss = self.cost_func(logits, labels[:,ii])
                    all_loss+=loss
                
                #get predicts
                if copy_control<copy_thres:
                    _, max_idxs = torch.max(score, dim=1)
                else:
                    _, max_idxs = torch.max(logits, dim=1)
                predicts.append(torch.unsqueeze(max_idxs, dim=0))
                
        predicts = torch.cat(predicts, dim=0)
        predicts = torch.transpose(predicts, 0, 1)
    
        if is_train:  #training
#             all_loss = torch.cat(all_loss, dim=1)
#             all_loss = torch.mean(all_loss, dim=1)
#             loss = torch.mean(all_loss)
            loss = all_loss/self.max_length
    
            #print('loss size: ', loss.size())
            #torch.cuda.empty_cache()
            if self.use_cuda:
                return loss, predicts.data.cpu().tolist()
            else:
                return loss, predicts.data.tolist()
        else:   #testing
            if self.use_cuda:
                return predicts.data.cpu().tolist()
            else:
                return predicts.data.tolist()
#         if is_train:  #training
#             if self.use_cuda:
#                 return all_loss/(self.max_length+1), predicts.data.cpu().numpy()
#             else:
#                 return all_loss/(self.max_length+1), predicts.data.numpy()
#         else:   #testing
#             if self.use_cuda:
#                 return predicts.data.cpu().numpy()
#             else:
#                 return predicts.data.numpy()
    
    
    def decode_topk_seqs(self, encoder, inputs, input_lens, topk=3):
        enc_outputs, (enc_hn, enc_cn) = encoder(inputs, input_lens)
        batch_size = enc_outputs.size(dim = 0)
        
        #one hot of inputs
        sents_len = enc_outputs.size(1)
        inputs = inputs[:,:sents_len].unsqueeze(dim=2)
        one_hot = torch.FloatTensor(batch_size, sents_len, len(self.vocab.word2token)).zero_()
        one_hot.scatter_(2, inputs, 1)
        one_hot = one_hot.transpose(1,2)
        one_hot = self._tocuda(Variable(one_hot, requires_grad = 0))
        
        metadata = self.decode_by_beamsearch(encoder_hidden=(enc_hn, enc_cn), encoder_outputs=enc_outputs, inputs_one_hot=one_hot,topk = topk)
        results = metadata['topk_sequence']
        results =torch.cat(results, dim = 2)
        results=results.view(batch_size*topk, -1)
        if self.use_cuda:
            results = results.data.cpu().tolist()
        else:
            results = results.data.tolist()
#         results=batch_tokens_remove_eos(results, self.vocab)

#         labels = [x for x in labels for ii in range(topk)]
#         labels = batch_tokens_remove_eos(labels, self.vocab)
#         bleu_scores = batch_tokens_bleu(references=labels, candidates=results, smooth_epsilon=0.01)
        
#         bleu_scores = torch.FloatTensor(bleu_scores).view(batch_size, topk)
#         bleu_max, _ = torch.max(bleu_scores, dim=1)
        
#         bleu_mean = torch.mean(bleu_scores, dim=1).unsqueeze(dim=1)
#         bleu_scores = bleu_scores-bleu_mean
#         bleu_scores = bleu_scores.view(-1)
        
#         bleu_scores = self._tocuda(Variable(bleu_scores, requires_grad = 0))
#         log_probs = metadata['score']
#         log_probs = log_probs.view(batch_size*topk)
#         loss = -torch.dot(log_probs, bleu_scores)/batch_size/topk
#         return loss, results, torch.mean(bleu_mean.squeeze()), torch.mean(bleu_max)

        log_probs = metadata['score']
        log_probs = log_probs.view(batch_size*topk)
        
        return results, log_probs
        
        
        
    def _tocuda(self, var):
        if self.use_cuda:
            return var.cuda()
        else:
            return var
    def decode_by_beamsearch(self, encoder_hidden=None, encoder_outputs=None, inputs_one_hot=None, topk = 10):
        self.k = topk
        batch_size = encoder_outputs.size(dim=0)
        
        self.pos_index = self._tocuda(Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1))

        hidden = tuple([_inflate(h, self.k, 1).view(batch_size*self.k, -1) for h in encoder_hidden])
        #print('hidden0 size: (%s, %s)'%(hidden[0].size(), hidden[1].size()))

        encoder_outputs = _inflate(encoder_outputs, self.k, 1).view(batch_size*self.k, encoder_outputs.size(1), encoder_outputs.size(2))
        inputs_one_hot = _inflate(inputs_one_hot, self.k, 1).view(batch_size*self.k, inputs_one_hot.size(1), inputs_one_hot.size(2))
        
        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = self._tocuda(Variable(sequence_scores))

        # Initialize the input vector
        input_var = self._tocuda(Variable(torch.LongTensor([self.SOS] * batch_size * self.k)))

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for ii in range(0, self.max_length):
            # Run the RNN one step forward
            #print('setp: %s'%ii)
            input_vec = self.embed(input_var)
            #print('input_var and input_vec size: ', input_var.size(), input_vec.size())
            
            hidden = self.lstmcell(input_vec, hidden)
            #print('hidden size: (%s, %s)'%(hidden[0].size(), hidden[1].size()))
            
            #get context vector
            context_vec, weight = self.get_context_vec(enc_outputs=encoder_outputs, this_timestep_input=-1, 
                                                            dec_state = hidden[0])
            
            logits = self.fcnn(torch.cat([hidden[0], context_vec], dim=1))
#             print('logits size', logits.size())
#             print(encoder_outputs.size())
#             print(input_vec.size())
#             print(hidden[0].size())
#             print(inputs_one_hot.size())
            if_copy, prob_copy = self.copy_mechanism(enc_outputs=encoder_outputs, this_timestep_input=input_vec.squeeze(dim=1), 
                                                     dec_state = hidden[0], inputs_one_hot = inputs_one_hot, 
                                                     context_v = context_vec, weight=weight)
#             print('if_copy size', if_copy.size(), 'prob_copy size', prob_copy.size())
            
            score = (1-if_copy)*self.softmax(logits)+if_copy*prob_copy
            score = torch.clamp(score, min=10**(-30), max=1)
#             print('score size: ', score.size())

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = _inflate(sequence_scores, self.V, 1)
            sequence_scores += torch.log(score).squeeze(1)
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)
            sequence_scores = scores.view(batch_size * self.k, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)
            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(0, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(0, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
#             stored_hidden.append(hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_size)

        metadata = {}

        metadata['score'] = s
        metadata['topk_length'] = l
        metadata['topk_sequence'] = p
        metadata['length'] = [seq_len[0] for seq_len in l]
        metadata['sequence'] = [seq[0] for seq in p]
        
#         torch.cuda.empty_cache()
        
        return metadata

    def _backtrack(self, hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.

        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state

        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.

            score [batch, k]: A list containing the final scores for all top-k sequences

            length [batch, k]: A list specifying the length of each sequence in the top-k candidates

            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        lstm = isinstance(hidden, tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = hidden[0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
        l = [[self.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]

                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)
            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.data[0]] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
#         output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        #    --- fake output ---
        output = None
        #    --- fake ---
        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
            score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)
            
            
class Seq2Seq(nn.Module):
    def __init__(self, use_cuda, input_dim, hidden_dim, vocab, max_length = 25):
        super(Seq2Seq, self).__init__()
        
        self.use_cuda = use_cuda
        self.enc = Encoder(use_cuda=use_cuda, hidden_dim=hidden_dim, input_dim=input_dim, vocab=vocab)
        self.dec = Decoder(use_cuda=use_cuda, encoder=self.enc, hidden_dim=hidden_dim, max_length=max_length)
        if use_cuda:
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
    def forward(self, inputs, input_lens, labels, is_train=1, teaching_rate=1):
        enc_outputs, (enc_hn, enc_cn) = self.enc(torch.LongTensor(inputs), torch.LongTensor(input_lens))
        if is_train:
            loss, predicts = self.dec(enc_outputs = enc_outputs, 
                                    h0_and_c0=(enc_hn, enc_cn), 
                                    sent_lens=input_lens,
                                    labels=torch.LongTensor(labels), 
                                    is_train=1, 
                                    teaching_rate = 1,
                                    inputs = inputs
                                    )
            return loss, predicts
        else:
            predicts = self.dec(enc_outputs = enc_outputs, 
                                h0_and_c0=(enc_hn, enc_cn), 
                                sent_lens=input_lens,
                                labels=torch.LongTensor(labels), 
                                is_train=0, 
                                teaching_rate = 1,
                                inputs = inputs
                                )
            return predicts
#     def train_using_rl(self, inputs, input_lens, labels, is_train=1, teaching_rate=1):
#         enc_outputs, (enc_hn, enc_cn) = self.enc(torch.LongTensor(inputs), torch.LongTensor(input_lens))
#         loss, predicts, bleu_mean = self.dec.train_using_rl_2(enc_outputs = enc_outputs, 
#                                                 h0_and_c0=(enc_hn, enc_cn), 
#                                                 sent_lens=input_lens,
#                                                 labels=labels,
#                                                 is_train=1, 
#                                                 teaching_rate = 1
#                                                 )
#         return loss, predicts, bleu_mean

    def tocuda(self, x):
        if self.use_cuda:
            return x.cuda()
        else:
            return x
        
    def train_using_reward(self, inputs, input_lens, reconstruct_labels, reconstruct_model, language_model, topk=3, loss_ratio=0.5):
        dec_seqs, log_probs = self.dec.decode_topk_seqs(self.enc, inputs, input_lens, topk=topk)
#         enc_outputs, (enc_hn, enc_cn) = self.enc(torch.LongTensor(inputs), torch.LongTensor(input_lens))
#         results = self.dec.decode_no_labels(enc_outputs=enc_outputs, h0_and_c0=(enc_hn, enc_cn), topk=topk)
        simple_sent1s, simple_sent2s = seqs_split(dec_seqs, self.enc.vocab)
        
        lm_input1s, lm_input1_lens, lm_label1s = get_lm_inputs_and_labels(simple_sent1s, self.enc.vocab, self.dec.max_length)
        simple_sent1s_ppl = language_model.get_sentences_ppl(torch.LongTensor(lm_input1s), 
                                                      torch.LongTensor(lm_input1_lens), 
                                                      torch.LongTensor(lm_label1s)
                                                    )
        lm_input2s, lm_input2_lens, lm_label2s = get_lm_inputs_and_labels(simple_sent2s, self.enc.vocab, self.dec.max_length)
        simple_sent2s_ppl = language_model.get_sentences_ppl(torch.LongTensor(lm_input2s), 
                                                      torch.LongTensor(lm_input2_lens), 
                                                      torch.LongTensor(lm_label2s)
                                                    )
        
        simple_inputs, simple_input_lens = simple_sents_concat(simple_sent1s, simple_sent2s, self.enc.vocab, self.dec.max_length)
        #reconstruct labels
        reconstruct_loss, predicts = reconstruct_model.forward(torch.LongTensor(simple_inputs), 
                                     torch.LongTensor(simple_input_lens), 
                                     labels=reconstruct_labels, 
                                     is_train=1, teaching_rate=1)
        
        #rm_rewards: reconstruct model rewards
        #lm_rewards: language model rewards
        rm_rewards=-reconstruct_loss.data
        lm_rewards=(1/self.tocuda(torch.Tensor(simple_sent1s_ppl))+1/self.tocuda(torch.Tensor(simple_sent2s_ppl)))/2
        
        rm_rewards_mean = torch.mean(rm_rewards.view(-1, topk), dim=1)
        lm_rewards_mean = torch.mean(lm_rewards.view(-1, topk), dim=1)
        rm_rewards = rm_rewards.view(-1, topk) - rm_rewards_mean.unsqueeze(dim=1)
        lm_rewards = lm_rewards.view(-1, topk) - lm_rewards_mean.unsqueeze(dim=1)
        
        rm_rewards = rm_rewards.view(-1)
        lm_rewards = lm_rewards.view(-1)
        
        #sum both rewards up
        rewards = loss_ratio*rm_rewards+(1-loss_ratio)*lm_rewards
        rewards = Variable(rewards, requires_grad=0)
        
        #regarding rewards as weights of every seq
        loss = -torch.dot(log_probs, rewards)/log_probs.size(0)
        
#         labels = [x for x in labels for ii in range(topk)]
#         labels = batch_tokens_remove_eos(labels, self.vocab)
#         bleu_scores = batch_tokens_bleu(references=labels, candidates=results, smooth_epsilon=0.01)
        
#         bleu_scores = torch.FloatTensor(bleu_scores).view(batch_size, topk)
#         bleu_max, _ = torch.max(bleu_scores, dim=1)
        
#         bleu_mean = torch.mean(bleu_scores, dim=1).unsqueeze(dim=1)
#         bleu_scores = bleu_scores-bleu_mean
#         bleu_scores = bleu_scores.view(-1)
        
#         bleu_scores = self._tocuda(Variable(bleu_scores, requires_grad = 0))
        
#         log_probs = metadata['score']
#         log_probs = log_probs.view(batch_size*topk)
    
#         loss = -torch.dot(log_probs, bleu_scores)/batch_size/topk
        
        return loss, reconstruct_loss, torch.mean(rm_rewards_mean), torch.mean(lm_rewards_mean)
    
    
