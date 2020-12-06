import os
import fnmatch
import sys
import json
from nblearn import load_tokens


def extract_test_data(path):
    test_files = []
    for root, _, filenames in os.walk(path):
        for file_name in fnmatch.filter(filenames, "*.txt"):
            test_files.append(os.path.join(root, file_name))
    return test_files


def classify(tokens, p_spam, p_ham, p_spam_token, p_ham_token):
    for t in tokens:
        if t in p_spam_token:
            p_spam += p_spam_token[t]
        # else:
            # p_spam += p_spam_token["<None>"]
        if t in p_ham_token:
            p_ham += p_ham_token[t]
        # else:
        #     p_ham += p_ham_token["<None>"]
    return p_spam > p_ham


def main():
    # with open('nbmodel_part2.txt', "rt", encoding='utf-8') as file:
    with open('nbmodel.txt', "rt", encoding='utf-8') as file:
        p_spam, p_ham, p_spam_token, p_ham_token = json.load(file)

    test_files = extract_test_data(sys.argv[1])
    output = ''

    for file in test_files:
        tokens = load_tokens(file)
        is_spam = classify(tokens, p_spam, p_ham, p_spam_token, p_ham_token)
        if is_spam:
            output += f"spam\t{file}\n"
        else:
            output += f"ham\t{file}\n"

    # with open('nboutput_part2.txt', 'wt') as file:
    with open('nboutput.txt', 'wt') as file:
        file.write(output)


if __name__ == '__main__':
    main()
