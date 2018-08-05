import sys
import os
import time
import random
import re
import json
import pickle
from typing import List, Tuple, Dict, Callable, Optional, Any, Sequence, Mapping, NamedTuple
import numpy as np

class PTBDataSource:
    
    def __init__(self, config: Config):
        self.config = config
        train_path = os.path.join(self.config.data_path, 'ptb.train.txt')
        test_path = os.path.join(self.config.data_path, 'ptb.test.txt')
        valid_path = os.path.join(self.config.data_path, 'ptb.valid.txt')
        
        self._word_to_id = self._create_tokenizer(train_path)
        self._id_to_word = {v: k for k, v in self._word_to_id.items()}
        
        self.train = self._create_data(train_path)
        self.test = self._create_data(test_path)
        self.valid = self._create_data(valid_path)
        
    def shuffle(self):
        random.shuffle(self.train)
        
    def feed_dict_list(self, model):
        num_batch = len(self.train) // self.config.batch_size
        batch_list = []
        inputs, inputs_length, target_ids = self._make_feed_list(self.train)
        
        # batch_sizeに分ける
        for i in range(num_batch):
            index_from = i * self.config.batch_size
            index_to = (i + 1) * self.config.batch_size
            batch_range = range(index_from, index_to)
            fd = {
                model.inputs: inputs[batch_range],
                model.inputs_length: inputs_length[batch_range],
                model.target_ids: target_ids[batch_range]1
            }
            batch_list.append(fd)
        return batch_list
    
    def feed_test_list(self, model):
        test_data = random.sample(self.test, 256)
        inputs, inputs_length, target_ids = self._make_feed_list(test_data)
        fd = {
            model.inputs: inputs,
            model.inputs_length: inputs_length,
            model.target_ids: target_ids
        }
        return fd

    def _make_feed_list(self, data):
        inputs = []
        inputs_length = []
        target_ids = []
        for (i, sentence) in enumerate(data):
            for j in range(len(sentence)-1):
                inputs_words = sentence[:j+1][-self.config.max_length:] 
                inputs.append(inputs_words + [self.pad_id] * (self.config.max_length - len(inputs_words)))
                inputs_length.append(len(inputs_words))
                target_ids.append(sentence[j+1])
        inputs = np.array(inputs)
        inputs_length = np.array(inputs_length)
        target_ids = np.array(target_ids)
        
        return inputs, inputs_length, target_ids

    def _read_all_words(self, path) -> List[str]:
        with open(path, 'r') as f:
            return f.read().replace('\n', '<eos>').split()
        
    def _read_sentences(self, path) -> List[List[str]]:
        with open(path, 'r') as f:
            sentences = f.read().split('\n')
            return [sentence.split() for sentence in sentences]

    def _create_tokenizer(self, path: str):
        data = self._read_all_words(path)
        counter = Counter(data)
        sorted_counter = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*sorted_counter))
        word_to_id = dict(zip(words, range(1, len(words)+1)))
        return word_to_id
        
    def _get_id_from_word(self, word: str) -> int:
        return self._word_to_id.get(word, self.unk_id)
    
    def _sentence_to_id_list(self, sentence: List[str]) -> List[int]:
        return [self._get_id_from_word(word) for word in sentence]
    
    def _create_data(self, path: str):
        return [self._sentence_to_id_list(sentence) for sentence in self._read_sentences(path)]
    
    @property
    def vocab_size(self) -> int:
        return len(self._word_to_id)
    
    @property
    def pad_id(self) -> int:
        return 0
    
    @property
    def unk_id(self) -> int:
        return self._word_to_id.get('<unk>', self.pad_id)
    
    @property
    def eos_id(self) -> int:
        return self._word_to_id.get('<eos>', self.pad_id)

