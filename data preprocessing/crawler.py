# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:57:24 2019

@author: THINKPAD
"""


from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import os 

path = os.getcwd()

dngs = []
tifs = []

main = "https://data.csail.mit.edu/graphics/fivek/"
page = urlopen(main)
soup = BeautifulSoup(page, 'html.parser')
for link in soup.findAll('a', href=True):
    href = link['href']
    if href.endswith('dng'):
        dngs.append(href)
    elif href.endswith('tif'):
        tifs.append(href)

# crawl dng
for i in range(500):
    url = dngs[i]
    name = url[8:13] + '.dng'
    urlretrieve(main + url, path + '/adobe5k/' + name)
    
# crawl tif
for i in range(2500):
    url = tifs[i]
    name = url[13:18] + url[11] + '.tif'
    urlretrieve(main + url, path + '/adobe5k/' + name)   

    