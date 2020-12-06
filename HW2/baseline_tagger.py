import sys
import pycrfsuite as suite
import hw2_corpus_tool as utils


def get_feature(dialog):
    items = []

    for index, utterance in enumerate(dialog):
        feature = []

        if not index:
            feature.append('First Utterance')

        if index and dialog[index-1].speaker != utterance.speaker:
            feature.append('Speaker Change')

        if utterance.pos:
            for token in utterance.pos:
                feature.append('TOKEN_' + token.token)
            for token in utterance.pos:
                feature.append('POS_' + token.pos)
        else:
            feature.append('NO_WORDS')
        items.append(feature)

    return items


def generate_features(dir_path):
    features = ([], [])
    for dialog in utils.get_data(dir_path):
        features[0].append(get_feature(dialog))
        features[1].append([utterance.act_tag for utterance in dialog])
    return features


def get_trainer(features):
    trainer = suite.Trainer(verbose=False)
    for xseq, yseq in zip(features[0], features[1]):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    return trainer


def train_model(dir_path):
    features = generate_features(dir_path)
    trainer = get_trainer(features)
    trainer.train(model_output)


def tag_model(dir_path, output_dir_path):
    features = generate_features(dir_path)

    tagger = suite.Tagger()
    tagger.open(model_output)

    with open(file=output_dir_path, mode='wt', encoding='utf-8') as file:
        for feature in features[0]:
            for tag in tagger.tag(feature):
                file.write(f'{tag}\n')
            file.write('\n')


def main():
    train_dir, test_dir, output_dir = sys.argv[1:4]
    train_model(train_dir)
    tag_model(test_dir, output_dir)


if __name__ == '__main__':
    model_output = 'baseline_model_crf.crfsuite'
    main()
