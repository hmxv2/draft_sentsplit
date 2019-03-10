import json
import pickle
import random
from collections import Counter

class Vocab:
    def __init__(self, stop_words_set=set()):
        self.word_counter=Counter()
        self.stopwords=stop_words_set
        
    def insert_sentence(self, sentence):#remove stop words?
        self.word_counter.update(sentence)
    
    def vocab_analyse(self, count_clip=5):
        removed_word_cnt=0
        removed_token_cnt=0

        all_word_cnt=0
        all_token_cnt=0
        
        for x in self.word_counter:
            all_word_cnt+=self.word_counter[x]
            all_token_cnt+=1
            
            if self.word_counter[x]<count_clip:
                removed_word_cnt+=self.word_counter[x]
                removed_token_cnt+=1
            
        print('in corpus, %2.4f%%(%s/%s) words will be removed if count_clip is %s.'%(100*removed_word_cnt/all_word_cnt, 
                                                                                      removed_word_cnt, all_word_cnt, count_clip))
        print('in vocabulary, %2.4f%%(%s/%s) tokens will be removed if count_clip is %s.'%(100*removed_token_cnt/all_token_cnt, 
                                                                                           removed_token_cnt, all_token_cnt, count_clip))
    
    def build_vocab(self, count_clip=0):
        self.token2word=['<sos>', '<padding>', '<eos>', '<low_freq>', '<mask>','<split>']
        self.word2token={'<sos>':0, '<padding>':1, '<eos>':2, '<low_freq>':3, '<mask>':4, '<split>':5}
        self.token2count={'<sos>':99, '<padding>':99, '<eos>':99, '<low_freq>':99, '<mask>':99, '<split>':99}
        
        for x in self.word_counter:
            if self.word_counter[x] >= count_clip:
                self.token2word.append(x)
                self.word2token[x]=len(self.token2word)-1
                self.token2count[x] = self.word_counter[x]
                
    def vocab_save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)