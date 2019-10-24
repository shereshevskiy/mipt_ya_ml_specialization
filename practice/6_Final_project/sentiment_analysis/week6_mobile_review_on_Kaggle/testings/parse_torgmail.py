import requests
import bs4
from multiprocessing import Pool
from functools import reduce
import pandas as pd

def parse_page(url):
    text = requests.get(url).text
    parser = bs4.BeautifulSoup(text, 'lxml')
    # парсим страницу
    page_reviews = [(item.find('a', attrs={'class':'more'}).attrs['full-text'] 
                     if item.find('a', attrs={'class':'more'}) else 
                     item.find('p', attrs={'class':"review-item__paragraph"}).text.replace('\n', '').replace('\t', ''), 
                     float(item.find('span', attrs={'class':"review-item__rating-counter"}).text.replace(',', '.')))
                    for item in  parser.find_all('div', attrs={'class':'review-item'})]
    return page_reviews

p = Pool(10)
n_start, n_end = 100, 901
url_list = ['https://torg.mail.ru/review/goods/mobilephones/?page=' + str(num) 
            for num in range(n_start, n_end)]
    
if __name__ == '__main__':    
    map_results = p.map(parse_page, url_list)
    reduce_results = reduce(lambda x,y: x + y, map_results)
    train_df = pd.DataFrame({'review':list(map(lambda x: x[0], reduce_results)), 
                          'rating':list(map(lambda x: x[1], reduce_results))})
    # подчистим текст
    train_df['review'] = train_df['review'].apply(lambda s: s.replace('\n', ' ').replace('\r', ' ').replace('<br />', ' ')
                                            .replace('\t', ' ').replace('&quot;', ' ').replace('.', ' ').replace(',', ' '))     
    
    #сохраним
    train_df.to_csv('data/train1_df.csv')