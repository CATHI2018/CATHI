'''
YUEXIAOLI---CATHI
'''

import os
import re
import sys

from tensorflow.python.platform import gfile
from configs.config import BUCKETS


_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")


def get_dialog_train_set_path(path):
    return os.path.join(path, 'chat')


def get_dialog_voc_set_path(path):
    return os.path.join(path, 'chatvoc')


def get_dialog_dev_set_path(path):
    return os.path.join(path, 'chat_test')


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):

    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
    print('total vaca size: %s' % len(vocab))
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + "\n")


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_dialog_data(data_dir, vocabulary_size):
    train_path = get_dialog_train_set_path(data_dir)
    dev_path = get_dialog_dev_set_path(data_dir)
    voc_path = get_dialog_voc_set_path(data_dir)


    vocab_path = os.path.join(data_dir, "vocab%d.in" % vocabulary_size)
    create_vocabulary(vocab_path, voc_path + ".in", vocabulary_size)

    train_ids_path = train_path + (".ids%d.in" % vocabulary_size)
    data_to_token_ids(train_path + ".in", train_ids_path, vocab_path)

    dev_ids_path = dev_path + (".ids%d.in" % vocabulary_size)
    data_to_token_ids(dev_path + ".in", dev_ids_path, vocab_path)

    return (train_ids_path, dev_ids_path, vocab_path)


def read_data(tokenized_dialog_path, max_size=None):
    data_set = [[] for _ in BUCKETS]
    with gfile.GFile(tokenized_dialog_path, mode="r") as fh:
        source, target = fh.readline(), fh.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()

            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(EOS_ID)

            for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            source, target = fh.readline(), fh.readline()
    return data_set
