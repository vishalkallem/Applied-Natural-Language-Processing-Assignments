import json
import sys
import math

from nblearn import NaiveBayesClassifier, load_tokens, search_train_files, get_count_tokens


def extract_train_data(dir_name='.'):
    nb_classifier.spam_files, nb_classifier.ham_files = search_train_files(dir_name, extension='*.txt')
    total_files = int(0.1 * (len(nb_classifier.spam_files) + len(nb_classifier.ham_files)))
    nb_classifier.spam_files = nb_classifier.spam_files[:total_files]
    nb_classifier.ham_files = nb_classifier.ham_files[:total_files]
    nb_classifier.spam_tokens = [t for path in nb_classifier.spam_files for t in load_tokens(path)]
    nb_classifier.ham_tokens = [t for path in nb_classifier.ham_files for t in load_tokens(path)]


def add_one_smoothing(tokens, denominator, label=1):
    p_token = {}
    for t in tokens:
        p_token[t] = math.log((tokens[t][label] + 1) / denominator, 10)
    # p_token["<None>"] = math.log(1/denominator, 10)
    return p_token


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
    with open('nbmodel_part2.txt', 'wt', encoding='utf-8') as file:
        json.dump([nb_classifier.p_spam, nb_classifier.p_ham, nb_classifier.p_spam_token, nb_classifier.p_ham_token],
                  file, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    nb_classifier = NaiveBayesClassifier()
    main()
