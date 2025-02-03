import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import nltk
import re

class Text_Classifer:
    def __init__(self):
        self.stop_words = {'aren', 'once', 'herself', "it's", 'what', 'whom', 'shouldn', 'have', 'now', 'most', 'no', 'her', 'own', 'itself', 've', "haven't", "mightn't", 'as', 'again', 'under', 'his', 'such', 'having', 'needn', 'had', 'a', 'in', 'were', 'when', 'y', 'doesn', 'an', 'more', 'but', "you're", 'all', 'into', 'or', 'for', 'ain', 'wasn', 'each', 'she', 'o', "isn't", 'before', 'is', "mustn't", "you'll", 'mightn', "doesn't", 'won', 'if', 'only', 'did', 'then', 'hadn', 'ma', "she's", 'with', "needn't", 'further', 'd', "couldn't", 'to', 'against', 'wouldn', 'couldn', 'does', 'same', 'themselves', 'both', 'should', 'weren', 'me', "weren't", 'down', "should've", 'yourself', "hadn't", 'the', 'because', 'during', 'about', 'below', 'will', "wouldn't", 'from', 'which', "didn't", 'my', 'than', "hasn't", 'above', 'him', 're', 'between', 'up', 'over', 'too', 'those', 'shan', 'm', 'hasn', 'doing', "shouldn't", 'these', 'few', 'out', 'yourselves', 'do', 'was', "won't", 'them', 'ourselves', 'i', 'there', 'can', 'we', 'yours', 'you', 'that', 'he', 'himself', 'being', 'so', 'they', 'didn', 'here', 'by', 'off', 'who', 'its', 'don', 'of', 'your', 'through', 'just', "aren't", 'their', 'some', 'nor', 'our', 'why', 'am', 'hers', 'at', 'until', "you've", 'other', 'ours', "you'd", 'on', 't', 'this', 'and', 'myself', 'after', 'it', 'very', 'any', "that'll", 'be', 'has', 'while', 's', "don't", 'not', 'theirs', 'how', 'haven', 'mustn', "wasn't", 'are', 'll', 'isn', 'where', 'been', "shan't"}
        self.topics = ['Legal', 'Governance', 'Social', 'Reputational Risk', 'Regulatory Risk', 'Physical Risk', 'Transition Risk', 'Waste and Water', 'Biodiversity', 'Fossil Fuels']

    def data_assigner(self, filename_work, filename_train):
        self.data_to_train = pd.read_csv(filename_train)
        self.data_to_evaluate = pd.read_csv(filename_work)["title"]
        self.data_to_train["processced_text"] = self.data_to_train["text"].apply(self.preprocess)
        self.data_to_evaluate["processced_text"] =  self.data_to_evaluate .apply(self.preprocess)

    def preprocess(self, data):
        processed_data = data.lower()
        processed_data = re.sub(r'[^a-zA-Z\s]', '', processed_data)
        processed_data = " ".join([word for word in processed_data.split() if word not in self.stop_words])
        return processed_data
    
    def classify_text(self):
        pass

if "__main__" == __name__:
    tc=Text_Classifer()
    tc.data_assigner("company_news.csv", "data.csv")
    tc.classify_text()
