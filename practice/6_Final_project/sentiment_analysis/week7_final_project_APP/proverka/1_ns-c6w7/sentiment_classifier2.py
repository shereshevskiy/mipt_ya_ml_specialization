# -*- coding: utf-8 -*-

import pickle

class SentimentClassifier(object):
    def __init__(self):
        with open("clf.dmp") as in_f:
            self.model = pickle.load(in_f)

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.25:
            return u"отрицательное"
        if probability < 0.5:
            return u"вероятно, отрицательное"
        if probability < 0.75:
            return u"вероятно, положительное"
        if probability < 1:
            return u"положительное"
        return ""

    def predict_text(self, text):
        return self.model.predict_proba([text])[0][1]

    def get_prediction_message(self, text):
        prediction_probability = self.predict_text(text)
        return self.get_probability_words(prediction_probability) + " (" + "{:2.2f}".format(prediction_probability*100) + "%)"
