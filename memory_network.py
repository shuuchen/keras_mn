from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import collections
import itertools
import nltk
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import plot_model

DATA_DIR = '/Users/shuchendu/Downloads/tasks_1-20_v1-2/en-10k/'
TRAIN_FILE = os.path.join(DATA_DIR, 'qa1_single-supporting-fact_train.txt')
TEST_FILE = os.path.join(DATA_DIR, 'qa1_single-supporting-fact_test.txt')

def get_data(file):
    stories, questions, answers = [], [], []
    story_text = []
    with open(file) as f:
        for line in f:
            line_no, text = line.split(' ', 1)
            if '\t' in text:
                question, answer, _ = text.split('\t')
                questions.append(question)
                answers.append(answer)
                stories.append(story_text)
                story_text = []
            else:
                story_text.append(text)
    f.close()

    return stories, questions, answers

def build_vocab(train_data, test_data):
    counter = collections.Counter()
    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            for sent in story:
                for word in nltk.word_tokenize(sent):
                    counter[word.lower()] += 1
        for question in questions:
            for word in nltk.word_tokenize(question):
                counter[word.lower()] += 1
        for answer in answers:
            for word in nltk.word_tokenize(answer):
                counter[word.lower()] += 1
    word2idx = {w: (i + 1) for i, (w, _) in enumerate(counter.most_common())}
    word2idx['PAD'] = 0
    idx2word = {v: k for k, v in word2idx.items()}

    return word2idx, idx2word

def get_maxlens(train_data, test_data):
    story_maxlen, question_maxlen = 0, 0
    for stories, questions, _ in [train_data, test_data]:
        for story in stories:
            story_len = 0
            for sent in story:
                story_len += len(nltk.word_tokenize(sent))
            story_maxlen = max(story_len, story_maxlen)
        for question in questions:
            question_len = len(nltk.word_tokenize(question))
            question_maxlen = max(question_len, question_maxlen)

    return story_maxlen, question_maxlen

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    X_story, X_question, Y = [], [], []
    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        X_s = [[word2idx[w.lower()] for w in nltk.word_tokenize(s)] for s in story]
        X_s = list(itertools.chain.from_iterable(X_s))
        X_q = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        X_story.append(X_s)
        X_question.append(X_q)
        Y.append(word2idx[answer.lower()])

    return pad_sequences(X_story, maxlen=story_maxlen), \
           pad_sequences(X_question, maxlen=question_maxlen), \
           np_utils.to_categorical(Y, num_classes=len(word2idx))



data_train = get_data(TRAIN_FILE)
data_test = get_data(TEST_FILE)
#print('train data\n{}'.format(data_train[:5]))
#print('test data\n{}'.format(data_test[:5]))

word2idx, idx2word = build_vocab(data_train, data_test)
#print('word: index\n{}'.format(word2idx))
#print('index: word\n{}'.format(idx2word))

vocab_size = len(word2idx)
#print('Vocab size: {}'.format(vocab_size))

story_maxlen, question_maxlen = get_maxlens(data_train, data_test)
#print('story maxlen: {}, question max len: {}'.format(story_maxlen, question_maxlen))

X_story_train, X_question_train, Y_train = vectorize(data_train, word2idx, story_maxlen, question_maxlen)
X_story_test, X_question_test, Y_test = vectorize(data_test, word2idx, story_maxlen, question_maxlen)
#print('shape of vectorized train data: {}, {}, {}'.format(X_story_train.shape, X_question_train.shape, Y_train.shape))

EMBEDDING_SIZE = 64
LATENT_SIZE = 32

# input
story_input = Input(shape=(story_maxlen, ))
question_input = Input(shape=(question_maxlen, ))

# story encoder memory
story_encoder = Embedding(input_dim=vocab_size,
                          output_dim=EMBEDDING_SIZE,
                          input_length=story_maxlen)(story_input)
story_encoder = Dropout(0.3)(story_encoder)

# question encoder
question_encoder = Embedding(input_dim=vocab_size,
                             output_dim=EMBEDDING_SIZE,
                             input_length=question_maxlen)(question_input)
question_encoder = Dropout(0.3)(question_encoder)

# match between story and question
match = dot([story_encoder, question_encoder], axes=[2, 2])

# encode story into vector space of question
story_encoder_c = Embedding(input_dim=vocab_size,
                            output_dim=question_maxlen,
                            input_length=story_maxlen)(story_input)
story_encoder_c = Dropout(0.3)(story_encoder_c)

# combine match and story vectors
response = add([match, story_encoder_c])
response = Permute((2, 1))(response)

# combine response and question vectors
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(LATENT_SIZE)(answer)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)
output = Activation('softmax')(answer)

model = Model(inputs=[story_input, question_input], outputs=output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# plot model
#plot_model(model, to_file='model.png', show_shapes=True)

BATCH_SIZE = 32
NUM_EPOCHES = 50
history = model.fit([X_story_train, X_question_train], [Y_train],
                    batch_size=BATCH_SIZE, epochs=NUM_EPOCHES,
                    validation_data=([X_story_test, X_question_test], [Y_test]))





