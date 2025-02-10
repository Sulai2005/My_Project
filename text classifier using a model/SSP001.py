import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
import nltk

class Text_Classifer:
    def __init__(self, filename_work, filename_train):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
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
