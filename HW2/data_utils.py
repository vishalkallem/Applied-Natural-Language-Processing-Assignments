import os
import random
import shutil
from baseline_tagger import generate_features


def get_file_paths():
    file_paths = []

    for root, _, files in os.walk(data_source):
        for file in files:
            if file.endswith('.csv'):
                file_paths.append((os.path.join(root, file), file))

    return file_paths


def split_train_dataset(file_paths):
    train_size = round(0.75 * len(file_paths))
    for index in range(train_size):
        shutil.copyfile(file_paths[index][0], data_dest + "\\train\\" + file_paths[index][1])
    return file_paths[train_size:]


def split_test_dataset(file_paths):
    for file in file_paths:
        shutil.copyfile(file[0], data_dest + "\\test\\" + file[1])


def create_directories():
    train_dir = data_dest + "\\train\\"
    test_dir = data_dest + "\\test\\"
    for path in [data_dest, train_dir, test_dir]:
        try:
            os.stat(path)
        except FileNotFoundError or FileExistsError or OSError:
            os.mkdir(path)


def split_data():
    create_directories()
    file_paths = get_file_paths()
    split_test_dataset(split_train_dataset(file_paths))


def evaluate_model(data_dest_path):
    test_dir = data_dest_path + "\\test\\"
    baseline_output_file = data_dest_path + "\\output_file.txt"
    advance_output_file = data_dest_path + "\\advanced_output_file.txt"
    features = generate_features(test_dir)

    for index, output_file in enumerate([baseline_output_file, advance_output_file]):
        with open(output_file, mode='rt', encoding='utf-8') as file:
            total, correct = 0, 0
            for actual_dialog in features[1]:
                for actual_tag in actual_dialog:
                    total += 1
                    if file.readline().strip() == actual_tag:
                        correct += 1
                file.readline()

            accuracy = correct/total
            print(f"{' Baseline Tagger ' if not index else ' Advanced Tagger '}".center(50, '*'))
            print(f"{'Correct':10}: {correct}")
            print(f"{'Total':10}: {total}")
            print(f"{'Accuracy':10}: {accuracy*100:10}\n\n")


if __name__ == '__main__':
    data_source = os.getcwd() + '\\train\\train\\train'
    data_dest = os.getcwd() + '\\split_data'
    # split_data()
    evaluate_model(data_dest)

