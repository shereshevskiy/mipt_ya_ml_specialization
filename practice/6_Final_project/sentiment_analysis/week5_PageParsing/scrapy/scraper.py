# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:10:00 2018

@author: dsher
"""

import scrapy
class BrickSetSpider(scrapy.Spider):
    name = "brickset_spider"
    start_urls = ['https://news.yandex.ru/yandsearch?cl4url=www.gazeta.ru/culture/2018/02/22/a_11659267.shtml&lang=ru&from=main_portal&stid=UmNmG1Amrxp5SKMx8uFX&t=1519284636&lr=193&msid=1519284963.90512.20956.55874&mlid=1519284636.glob_225.06e6bec3']
    def parse(self, response):
        SET_SELECTOR = '.set'
        for brickset in response.css(SET_SELECTOR):
            NAME_SELECTOR = 'h1 a ::text'
            yield {
            'name': brickset.css(NAME_SELECTOR).extract_first(),
            }