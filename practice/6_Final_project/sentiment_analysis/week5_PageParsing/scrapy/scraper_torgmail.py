# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:10:00 2018

@author: dsher
"""

import scrapy
class TorgmailSpider(scrapy.Spider):
    name = "torgmail_spider"
    start_urls = ['https://torg.mail.ru/review/goods/mobilephones/']
    def parse_torgmail_to_csv(url_list, path_and_filename='data/tmp.csv'):
        with open(path_and_filename, 'w+') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['review', 'rating']) # пишем заголовки колонок
            
            for url in tqdm_notebook(url_list, total=len(url_list)):
    #             request = requests.get(url)
                
                # если возникает ошибка (страница не найдена (404) или запрос прерван (443) или еще что-то)
                # то генерируется исключение и печатается сообщение об этом, но код не прерывается, а выполняется дальше
                # в "тексте" при этом генерируется пустота (?) и записывается она как NaN (его потом надо будет отфильтровать)
                try:                              # обрабатываем возможные исключения
                    request = requests.get(url)
                    request.raise_for_status()
                except requests.exceptions.HTTPError as err:
    #                 print('Oops. HTTP Error occured. Response is: {content} for url {url}'.format(content=err.response, url=url))
    #                 print('Response is: {content}'.format(content=err.response.content))
                    print('Oops. HTTP Error occured: %s' % err)
                except requests.exceptions.ConnectionError as err1:
    #                 print('ERROR: %s' % err1.args[0])
                    print('Oops. Connection Error occured. Response is: {content} for url {url}'.format(content=err1.response, 
                                                                                                        url=url))
                except requests.exceptions.RequestException as err2:
    #                 print('ERROR: %s' % err2.args[0])
                    print('Oops. RequestException Error occured. Response is: {content} for url {url}'.format(content=err2.response, 
                                                                                                        url=url))
                else:
                    text = request.text
                    parser = bs4.BeautifulSoup(text, 'lxml')    
    
                    for item in  parser.find_all('div', attrs={'class':'review-item'}):
    
    
                        review = item.find('a', attrs={'class':'more'}).attrs['full-text'] \
                                        if item.find('a', attrs={'class':'more'}) \
                                        else item.find('p', attrs={'class':"review-item__paragraph"}).text
                        rating = float(item.find('span', attrs={'class':"review-item__rating-counter"}).text.replace(',', '.'))
    
                        try:
                            csvwriter.writerow([review, rating])
                        except Exception as err3:
                            print('Oops. Exception Error occured for url {url}. Response is: {content}'.format(url=url, 
                                                                                                               content=err3))    
       
    pass
            NEXT_PAGE_SELECTOR = '.next a ::attr(href)'
            next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()
            if next_page:
                yield scrapy.Request(
                response.urljoin(next_page),
                callback=self.parse
                )