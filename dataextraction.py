import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, NoSuchAttributeException, ElementNotInteractableException
import requests
import requests_cache
import json
import time
import random
import os
from pprint import pprint
#from fnmatch import fnmatch

with open('json/urls.json') as f:
    urls = json.load(f)
    f.close()

download_path =  "/Users/drilonpollozhani/Code projects/Movie recommendations/data/raw"

def chrome(quiet=True):
    path = "/Users/drilonpollozhani/Documents/WebDriver/chromedriver"
    opt = None

    if quiet:
        opt = webdriver.ChromeOptions()
        opt.add_argument('--headless')
        opt.add_experimental_option("prefs", {
             "download.default_directory": download_path,
             "download.prompt_for_download": False
             }
        )

        browser = webdriver.Chrome(options=opt, executable_path=path)

        #browser.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        #params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_path}}
        #browser.execute("send_command", params) 
    else:
        browser = webdriver.Chrome(executable_path=path)

    return browser

def download_own_ratings(browser): #ADD ALSO OWN WATCHLIST IF/WHEN POSSIBLE
    assert isinstance(browser, selenium.webdriver.remote.webdriver.WebDriver), 'broswer must be a selenium.webdriver instance!'
    
    # Go to url with browser
    url = urls['imdb personal']['logon']
    browser.get(url)
    
    # Get logon credentials
    username, password = os.environ.get('IMDB_USER'), os.environ.get('IMDB_PWD')

    # Submit credentials and login
    email_input = browser.find_element_by_name('email')
    password_input = browser.find_element_by_name('password')
    login = browser.find_element_by_id('signInSubmit')
    remember = browser.find_element_by_name('rememberMe')

    email_input.send_keys(username)
    password_input.send_keys(password)
    remember.click()
    time.sleep(random.random())
    login.click()
    time.sleep(1)
    
    #Go to ratings page
    ratings_page = urls['imdb personal']['ratings2']
    browser.get(ratings_page)
    time.sleep(3)
    
    #Find export link, with 5 retries
    print('Downloading own ratings...')
    for i in range(10,1,-1): 
        try:
            export = browser.find_element_by_xpath('//ul[@class="pop-up-menu-list-items"]/li[1]/a')
            download_link = export.get_attribute('href')
            browser.get(download_link)
            time.sleep(1)
            browser.quit()
        except (NoSuchElementException, NoSuchAttributeException) as e:
            print(f'Error found: {e}...\nWaiting 15 seconds and retrying {i-1} more times.')
            time.sleep(15)
        else:
            break

def download_movie_data():
    ''' Chunkwise downloads and saves files from IMDB public repository '''
    with requests.Session() as session:
        for filename, url in urls['imdb general'].items():
            requests_cache.install_cache('request_caches/imdb_cache', backend='sqlite', expire_after=3600)
            response = session.get(url, stream=True)
            if response.status_code == 200:
                print(f'Downloading {filename}...')
                with open(f"data/raw/{filename}.tsv.gz", 'wb') as f:
                    for content in response.iter_content(chunk_size=50*10**6):
                        if content:
                            f.write(content)
                f.close()    
        session.close()


if __name__ == '__main__': 
    download_own_ratings(chrome())
    #download_movie_data()