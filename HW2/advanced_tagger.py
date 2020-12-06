import sys
import pycrfsuite as suite
import hw2_corpus_tool as utils


def get_feature(dialog):
    items = []

    for index, utterance in enumerate(dialog):
        feature = [utterance.text]

        if not index:
            feature.append('First Utterance')
        else:
            feature.append('Any Other Utterance')

        if index and dialog[index - 1].speaker != utterance.speaker:
            feature.append('Speaker Change')

        if '?' in utterance.text:
            feature.append('Question')

        if utterance.pos:
            feature.append('FIRST_TOKEN_' + utterance.pos[0].token)
            feature.append('LAST_TOKEN_' + utterance.pos[-1].token)
            feature.append('FIRST_POS_TAG_' + utterance.pos[0].pos)
            feature.append('LAST_POS_TAG_' + utterance.pos[-1].pos)
            tokens, pos_tags, token_bigrams, pos_tag_bigrams = [], [], [], []
            for idx, token in enumerate(utterance.pos):
                feature.append('TOKEN_' + token.token)
                pos_tags.append(token.pos)
                if idx + 1 < len(utterance.pos):
                    token_bigrams.append('TOKEN_BIGRAM_' + token.token + '-' + utterance.pos[idx+1].token)
                    pos_tag_bigrams.append('POS_TAG_BIGRAM_' + token.pos + '-' + utterance.pos[idx+1].pos)

            feature.extend(pos_tags)
            feature.extend(token_bigrams)
            feature.extend(pos_tag_bigrams)
            feature.append('LENGTH_' + str(len(utterance.pos)))
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
    model_output = 'advanced_model_crf.crfsuite'
    main()
