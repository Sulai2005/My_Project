from requests.exceptions import HTTPError
import pandas as pd
from GoogleNews import GoogleNews
import time

class news_scrapper:
    def __init__(self, filename = None):
        self.datas = pd.read_csv(filename)
        companies = []
        for index,row in self.datas.iterrows():
            company= row['company']
            companies.append(company)
        self.list_of_companies = companies
    def fetch_news(self):
        data = self.list_of_companies
        gn = GoogleNews(start="08-01-2024",end="12-01-2024")
        self.news = []
        counter = 0
        try:
            for company in data:
                gn.search(company)
                for res in gn.result():
                    self.news.append({
                        "company" : company,
                        "title" : res.get('title'),
                        "link" : res.get('link')
                    })
                gn.clear()
                time.sleep(5)
                counter += 1
                if counter == 3:
                    break
        except HTTPError as e:
            print("Error Encountered : ", e)
            time.sleep(10)

    def store_data(self, filename = "cn_o1.csv"):
        data_conv = pd.DataFrame(self.news)
        data_conv.to_csv(filename, encoding="utf-8", index= False)
        print('file created')

if __name__ =="__main__":
    ns = news_scrapper("fortune500-2019.csv")
    ns.fetch_news()
    ns.store_data("company_news.csv")

