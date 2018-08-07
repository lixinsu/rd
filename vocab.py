# coding = utf-8
# author = xy

import pickle


class Vocab:
    """ establish a relationship between word and index """

    def __init__(self, name='en'):
        self.name = name
        self.init_list = ['<pad>', '<s>', '<eos>', '<unk>']

        self.w2i = {}
        self.i2w = {}
        self.w2c = {}
        self.count = 0

        for word in self.init_list:
            self.w2i[word] = self.count
            self.i2w[self.count] = word
            self.w2c[word] = 1
            self.count += 1

    def add(self, words_list):
        """
        add list of words to vocab
        :param words_list:  list
        :return: .
        """
        assert isinstance(words_list, list)
        for word in words_list:
            if word in self.w2i:
                self.w2c[word] += 1
            else:
                self.w2i[word] = self.count
                self.i2w[self.count] = word
                self.w2c[word] = 1
                self.count += 1

    def filter(self, max_size):
        """
        filter vocab based on max_size
        :param max_size: int
        :return: .
        """
        print('The size of original vocab is %d.' % len(self.w2i))
        if len(self.w2i) > max_size:
            items = sorted(self.w2c.items(), key=lambda p: p[1], reverse=True)[: max_size-len(self.init_list)]
            self.w2i = {}
            self.i2w = {}
            self.count = 0

            for word in self.init_list:
                self.w2i[word] = self.count
                self.i2w[self.count] = word
                self.count += 1

            for w, c in items:
                if w not in self.w2i:
                    self.w2i[w] = self.count
                    self.i2w[self.count] = w
                    self.count += 1
                else:
                    continue
        print('The size of filtered vocab is %d.' % len(self.w2i))

    def words2indexes(self, word_list):
        """
        words 2 indexes
        :param word_list: words_list
        :return: indexes_list
        """
        def word2index(word):
            if word in self.w2i:
                return self.w2i[word]
            else:
                return 3  # <unk> --> 3

        return [word2index(word) for word in word_list]

    def indexes2words(self, index_list):
        """
        indexes 2 words
        :param index_list: indexes_list
        :return: words_list
        """
        def index2word(index):
            if index in self.i2w:
                return self.i2w[index]
            else:
                return '<unk>'

        return [index2word(index) for index in index_list]

    def save(self, file_path):
        vocab = {'w2i': self.w2i, 'i2w': self.i2w}
        with open(file_path, 'wb') as file:
            pickle.dump(vocab, file)

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            lang = pickle.load(file)
        self.w2i = lang['w2i']
        self.i2w = lang['i2w']
