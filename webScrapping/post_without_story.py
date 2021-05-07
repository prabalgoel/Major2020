# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 08:01:49 2020

@author: shash
"""
"""
//*[@id="react-root"]/section/main/div/div[2]/article/div[1]/div/div[1]

"""



import requests
from bs4 import BeautifulSoup
import time

import pandas as pd
df = pd.DataFrame(columns =['likes','comments'])

url="https://www.instagram.com/rosannapansino/"
number_of_post=350

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

driver=webdriver.Chrome(executable_path="C:\\Program Files\\chromeDriverSelenium\\chromedriver.exe")
driver.get(url)
driver.implicitly_wait(10)
print(driver.title)
content = driver.find_element_by_xpath('/html/body/div[1]/section/main/div/div[2]/article/div[1]/div/div[1]/div[1]/a')
content.click()

username=driver.find_element_by_xpath('/html/body/div[5]/div[2]/div/div[2]/div/div/div[1]/div/form/div[1]/div[1]/div/label/input')
username.send_keys("")


password=driver.find_element_by_name('password')

password.send_keys("")   

button=driver.find_element_by_xpath('/html/body/div[5]/div[2]/div/div[2]/div/div/div[1]/div/form/div[1]/div[3]/button')
button.send_keys(Keys.RETURN)

#print("done")
driver.implicitly_wait(20)

search_bar=driver.find_element_by_xpath('/html/body/div[1]/section/main/div/div/div/section/div/button')
print(search_bar)
search_bar.click()

prevx=20
prevy=526
name=driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/header/section/div[1]/h2')
print(name.text)
try:
    for i in range(1,(int)((number_of_post+3-1)/3)):
        #print(df.head())
        jj=i
        print("row number ",i)
        driver.execute_script("""var start = arguments[0],end=arguments[1] ;window.scroll(start,end);""",prevx,prevy)
        prevy+=276
        time.sleep(0.5)
        if(jj>=10):jj=10
        for j in range(1,4):
            #/html/body/div[1]/section/main/div/div[2]/article/div[1]/div/div[7]/div[1]
            path='/html/body/div[1]/section/main/div/div[2]/article/div[1]/div/div['+str(jj)+']/div['+str(j)+']/a'
            #print(path)
            content=driver.find_element_by_xpath(path)
            location = content.location
            #print(location)
            action = ActionChains(driver)
            action.move_to_element(content).perform()
            html = driver.page_source
            soup = BeautifulSoup(html, "lxml")
            mydivs=driver.find_element_by_class_name('_2z6nI')
            #mydivs=driver.find_element_by_css_selector('div')
            likes=0
            comments=0
            try:
                detail=mydivs.text
                print(type(detail),detail.split('\n'))
                likes=detail.split('\n')[0]
                comments=detail.split('\n')[1]
                to_append = [likes, comments]
                df_length = len(df)
                df.loc[df_length] = to_append
            except:
                to_append = [likes, comments]
                df_length = len(df)
                df.loc[df_length] = to_append
    
finally:        
    df.to_csv('C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping\\counts\\rosannapansino.csv',index=False)        
    driver.close() 
