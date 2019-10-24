__author__ = 'dmitry_sh'
from sklearn.externals import joblib
#import pickle as pkl
import os

abs_path = 'C:/Users/dsher/OneDrive/Documents/!_SkillsEvolution/ML/\
mipt_ya_ml_spec/practice/6_Final_project/sentiment_analysis/week4_flask_app/sentiment_demo_py'
# приходится использовать абсолютный путь, потому что в консоле на винде
# относительный путь не "прокатывает", как в ноутбуке или в отладчике на Spyder

path_to_data = ('data' )


class SentimentClassifier(object):
    def __init__(self):
        #with open(os.path.join(PATH_TO_DATA, 'model.jbl'), 'rb') as file:
        #    self.model = pkl.load(file)
        self.model = joblib.load(os.path.join(abs_path, path_to_data, 'model.jbl'))
        self.vectorizer = joblib.load(os.path.join(abs_path, path_to_data, 'vectorizer.jbl'))
        self.classes_dict = {0: "negative", 
                             1: "positive", 
                            -1: "prediction error"}


    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0],\
                   self.model.predict_proba(vectorized)[0].max()
        except:
            print ("prediction error")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print ('prediction error')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]