import sys

# True Positive - Spam Spam
# True Negative - Ham Ham
# False Positive - Spam Ham
# False Negative - Ham Spam

# Confusion Matrix
# Actual/Predicted  SPAM    HAM
# SPAM              TP      FN
# HAM               FP      TN

# Precision = True Positive / (True Positive + False Positive)
# Recall =  True Positive / (True Positive + False Negative)
# F1 score = 2 * (( Precision * Recall ) / ( Precision + Recall ))
# Accuracy = (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)


class ConfusionMatrix:
    def __init__(self):
        self.true_positive, self.true_negative, self.false_positive, self.false_negative = 0, 0, 0, 0
        self.spam_precision, self.ham_precision = None, None
        self.spam_recall, self.ham_recall = None, None
        self.spam_f1_score, self.ham_f1_score = None, None
        self.accuracy = None


def calculateConfusionMatrix():
    with open(sys.argv[1], mode="rt", encoding='utf-8') as file:
        data = file.read()
        for line in data.split('\n')[:-1]:
            label, path = line.split('\t')
            if label == "spam":
                if label in path:
                    cm.true_positive += 1
                else:
                    cm.false_positive += 1
            else:
                if label in path:
                    cm.true_negative += 1
                else:
                    cm.false_negative += 1


def calculateAccuracy():
    cm.accuracy = (cm.true_negative + cm.true_positive) / \
                  (cm.true_negative + cm.true_positive + cm.false_negative + cm.false_positive)


def calculatePrecision():
    cm.spam_precision = cm.true_positive / (cm.true_positive + cm.false_positive)
    cm.ham_precision = cm.true_negative / (cm.true_negative + cm.false_negative)


def calculateRecall():
    cm.spam_recall = cm.true_positive / (cm.true_positive + cm.false_negative)
    cm.ham_recall = cm.true_negative / (cm.true_negative + cm.false_positive)


def calculateF1Score():
    cm.spam_f1_score = 2 * ((cm.spam_precision * cm.spam_recall)/(cm.spam_precision + cm.spam_recall))
    cm.ham_f1_score = 2 * ((cm.ham_precision * cm.ham_recall)/(cm.ham_precision + cm.ham_recall))


if __name__ == '__main__':
    cm = ConfusionMatrix()
    calculateConfusionMatrix()
    calculateAccuracy()
    calculatePrecision()
    calculateRecall()
    calculateF1Score()
    print(f"Accuracy: {cm.accuracy}\n\nSpam Precision: {cm.spam_precision}\nHam Precision: {cm.ham_precision}\n\n")
    print(f"Spam Recall: {cm.spam_recall}\nHam Recall: {cm.ham_recall}\n\n")
    print(f"Spam F1 Score: {cm.spam_f1_score}\nHam F1 Score: {cm.ham_f1_score}")
