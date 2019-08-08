import numpy as np


def get_word_count(word_count_path):

    word2count = {}

    with open(word_count_path, 'r') as f:

        for i in f:
            i = i.strip()

            if len(i) > 0:
                i = i.split()

                if len(i) == 2:
                    word2count[i[0]] = float(i[1])
                else:
                    print(i)

    return word2count


def get_word_map(wv_path):

    words = []
    we = []

    with open(wv_path, 'r') as f:
        lines = f.readlines()

        for n, line in enumerate(lines):

            if n == 0:
                print(line)
                continue

            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            we.append(vect)
            words.append(word)

    return words, we


def read_eval_index_dataset(data_path, is_label=True):

    sentences = []
    mask_words = []
    mask_labels = []

    with open(data_path, "r", encoding='ISO-8859-1') as reader:

        while True:
            line = reader.readline()

            if not line:
                break

            sentence, words = line.strip().split('\t', 1)
            mask_word, labels = words.strip().split('\t', 1)

            label = labels.split('\t')

            sentences.append(sentence)
            mask_words.append(mask_word)

            one_labels = []
            for la in label[1:]:
                if la not in one_labels:
                    la_id, la_word = la.split(':')
                    one_labels.append(la_word)

                    # print(mask_word, " ---",one_labels)
            mask_labels.append(one_labels)

    return sentences, mask_words, mask_labels


def read_eval_dataset(data_path, is_label=True):
    sentences = []
    mask_words = []
    mask_labels = []
    id = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if id == 1:
                    continue
                if not line:
                    break
                sentence, words = line.strip().split('\t', 1)
                # print(sentence)
                mask_word, labels = words.strip().split('\t', 1)
                label = labels.split('\t')

                sentences.append(sentence)
                mask_words.append(mask_word)

                one_labels = []
                for la in label:
                    if la not in one_labels:
                        one_labels.append(la)

                # print(mask_word, " ---",one_labels)

                mask_labels.append(one_labels)
            else:
                if not line:
                    break
                # print(line)
                sentence, mask_word = line.strip().split('\t')
                sentences.append(sentence)
                mask_words.append(mask_word)
    return sentences, mask_words, mask_labels


def read_file(input_file):
    """
        Read a list of `InputExample`s from an input file.
    """
    sentences = []

    with open(input_file, "r", encoding='utf-8') as reader:
        while True:

            line = reader.readline()

            if not line:
                break

            line = line.strip()
            sentences.append(line)

    return sentences
