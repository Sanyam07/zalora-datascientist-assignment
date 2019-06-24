# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def score(y_true, y_pred):
    female_rate = sum(y_pred[y_true]) / sum(y_true)
    male_rate = sum(~y_pred[~y_true]) / sum(~y_true)
    return (female_rate + male_rate) / 2

def print_score(y_true, y_pred):
    print("Evalute based on test set")
    print(" - custom score: " + " %s" % score(y_true, y_pred))
    print(" - f1 micro: " + " %s" % f1_score(y_true, y_pred, average = 'micro'))
    print(" - f1 macro: " + " %s" % f1_score(y_true, y_pred, average = 'macro'))
    print(" - accuracy score" + " %s" % accuracy_score(y_true, y_pred))
    print(" - confusion matrix: ")
    print(confusion_matrix(y_true, y_pred))
