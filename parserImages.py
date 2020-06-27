'''
import urllib.request as url_

headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
      }
url = 'https://www.pinterest.ru/search/pins/?q=lofi%20art&rs=typed&term_meta[]=lofi%7Ctyped&term_meta[]=art%7Ctyped'
r = url_.urlopen("https://vk.com")

with open('test.html', 'wb') as output:
    output.write(r.read())

'''

import requests
import json
from bs4 import BeautifulSoup
import re


def get_joke():
    url = 'https://www.pinterest.ru/pin/784048616359322226/'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    joke = soup.find("script", id="initial-state", type="application/json")
    print(str(joke)[51:-9])
    x = json.loads(str(joke)[51:-9])
    print(x["resourceResponses"][0]["response"]["data"]["images"])
  #  badSymbols = re.findall(r'<[^>]*>', joke)
   # for c in badSymbols:
    #    joke = joke.replace(c, ' ')
    #joke = joke.replace('"', '')
    #joke = joke.replace('«', '')
    #joke = joke.replace('»', '')
    #return joke

get_joke()