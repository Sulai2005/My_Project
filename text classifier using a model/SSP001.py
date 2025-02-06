import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import re

class Text_Classifer:
    def __init__(self, filename_work, filename_train):
        self.stop_words = {'aren', 'once', 'herself', "it's", 'what', 'whom', 'shouldn', 'have', 'now', 'most', 'no', 'her', 'own', 'itself', 've', "haven't", "mightn't", 'as', 'again', 'under', 'his', 'such', 'having', 'needn', 'had', 'a', 'in', 'were', 'when', 'y', 'doesn', 'an', 'more', 'but', "you're", 'all', 'into', 'or', 'for', 'ain', 'wasn', 'each', 'she', 'o', "isn't", 'before', 'is', "mustn't", "you'll", 'mightn', "doesn't", 'won', 'if', 'only', 'did', 'then', 'hadn', 'ma', "she's", 'with', "needn't", 'further', 'd', "couldn't", 'to', 'against', 'wouldn', 'couldn', 'does', 'same', 'themselves', 'both', 'should', 'weren', 'me', "weren't", 'down', "should've", 'yourself', "hadn't", 'the', 'because', 'during', 'about', 'below', 'will', "wouldn't", 'from', 'which', "didn't", 'my', 'than', "hasn't", 'above', 'him', 're', 'between', 'up', 'over', 'too', 'those', 'shan', 'm', 'hasn', 'doing', "shouldn't", 'these', 'few', 'out', 'yourselves', 'do', 'was', "won't", 'them', 'ourselves', 'i', 'there', 'can', 'we', 'yours', 'you', 'that', 'he', 'himself', 'being', 'so', 'they', 'didn', 'here', 'by', 'off', 'who', 'its', 'don', 'of', 'your', 'through', 'just', "aren't", 'their', 'some', 'nor', 'our', 'why', 'am', 'hers', 'at', 'until', "you've", 'other', 'ours', "you'd", 'on', 't', 'this', 'and', 'myself', 'after', 'it', 'very', 'any', "that'll", 'be', 'has', 'while', 's', "don't", 'not', 'theirs', 'how', 'haven', 'mustn', "wasn't", 'are', 'll', 'isn', 'where', 'been', "shan't"}
        self.topics = ['Legal', 'Governance', 'Social', 'Reputational Risk', 'Regulatory Risk', 'Physical Risk', 'Transition Risk', 'Waste and Water', 'Biodiversity', 'Fossil Fuels']
        self.filename_train = filename_train
        self.filename_work = filename_work
        self.data_assigner()

    def data_assigner(self):
        self.data_to_train = pd.read_csv(self.filename_train)
        self.data_to_train["processced_text"] = self.data_to_train["text"].apply(self.preprocess)
        self.data_to_train['label'] = self.data_to_train['topic'].apply(lambda x: self.topics.index(x))
        self.data_to_evaluate = pd.read_csv(self.filename_work)
        self.data_to_evaluate["processced_text"] =  self.data_to_evaluate["text"].apply(self.preprocess)

    def preprocess(self, data):
        processed_data = data.lower()
        processed_data = re.sub(r'[^a-zA-Z\s]', '', processed_data)
        processed_data = " ".join([word for word in processed_data.split() if word not in self.stop_words])
        return processed_data
    
    def vectorize(self, data):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        x = self.vectorizer.fit_transform(data).toarray()
        return x
    
    def train_data(self):
        x_vectorized = self.vectorize(self.data_to_train['processced_text'])
        self.model = LogisticRegression()
        self.model.fit(x_vectorized, self.data_to_train['label'])

    def classify_text(self):
        self.train_data()
        y_vectorized = self.vectorizer.transform(self.data_to_evaluate['processced_text'])
        pred = self.model.predict(y_vectorized)
        self.data_to_evaluate['predicted'] = pred

    def output(self, filename: str):
        self.data_to_evaluate[['text', 'predicted']].to_csv(filename, index=False, encoding='utf-8')
        print("file created")
if "__main__" == __name__:
    tc=Text_Classifer("data.csv", "data1.csv")
    tc.classify_text()
    tc.output("output.csv")
