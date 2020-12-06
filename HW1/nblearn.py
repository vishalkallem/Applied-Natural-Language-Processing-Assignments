import os
import fnmatch
import math
import json
import sys
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.p_spam, self.p_ham = 0.0, 0.0
        self.spam_files, self.ham_files = None, None
        self.spam_tokens, self.ham_tokens = None, None
        self.p_spam_token, self.p_ham_token = None, None


def load_tokens(path):
    tokens = []
    with open(path, mode='r', encoding='latin1') as file:
        data = file.read().lower()
        for line in data.split('\n'):
            # words = ([t for t in line.split() if len(t) > 2])
            # for i in range(len(words) - 2 + 1):
            #     tokens += [' '.join(words[i:i + 1])]
            tokens.extend(line.split())
    return tokens


def search_train_files(dir_name='.', extension=''):
    train_spam_files, train_ham_files = [], []
    for root, _, filenames in os.walk(dir_name):
        for file_name in fnmatch.filter(filenames, extension):
            if "spam" in file_name:
                train_spam_files.append(os.path.join(root, file_name))
            elif "ham" in file_name:
                train_ham_files.append(os.path.join(root, file_name))
    return train_spam_files, train_ham_files


def extract_train_data(dir_name='.'):
    nb_classifier.spam_files, nb_classifier.ham_files = search_train_files(dir_name, extension='*.txt')
    nb_classifier.spam_tokens = [t for path in nb_classifier.spam_files for t in load_tokens(path)]
    nb_classifier.ham_tokens = [t for path in nb_classifier.ham_files for t in load_tokens(path)]


def add_one_smoothing(tokens, denominator, label=1):
    p_token = {}
    for t in tokens:
        p_token[t] = math.log((tokens[t][label] + 1) / denominator, 10)
    # p_token["<None>"] = math.log(1/denominator, 10)
    return p_token


# def get_token_count(tokens):
#     count_d = {}
#     for t in tokens:
#         count_d[t] = count_d.get(t, 0) + 1
#     return count_d


def get_count_tokens(spam_tokens, ham_tokens):
    count_d = defaultdict(lambda: [0, 0])
    for t in spam_tokens:
        count_d[t][1] += 1
    for t in ham_tokens:
        count_d[t][0] += 1
    return count_d


def calculate_probabilities(spam_data_files, ham_data_files):
    spam_files_len, ham_files_len = len(spam_data_files), len(ham_data_files)
    total_files = spam_files_len + ham_files_len
    nb_classifier.p_spam = math.log(spam_files_len / total_files, 10)
    nb_classifier.p_ham = math.log(ham_files_len / total_files, 10)
    count_d = get_count_tokens(nb_classifier.spam_tokens, nb_classifier.ham_tokens)
    unique_tokens = len(count_d)
    nb_classifier.p_spam_token = add_one_smoothing(count_d,
                                                   len(nb_classifier.spam_tokens) + unique_tokens)
    nb_classifier.p_ham_token = add_one_smoothing(count_d,
                                                  len(nb_classifier.ham_tokens) + unique_tokens, 0)


def main():
    extract_train_data(sys.argv[1])
    calculate_probabilities(nb_classifier.spam_files, nb_classifier.ham_files)
    with open('nbmodel.txt', 'wt', encoding='utf-8') as file:
        json.dump([nb_classifier.p_spam, nb_classifier.p_ham, nb_classifier.p_spam_token, nb_classifier.p_ham_token],
                  file, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    nb_classifier = NaiveBayesClassifier()
    main()
