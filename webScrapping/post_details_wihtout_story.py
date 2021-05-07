# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 08:40:51 2020

@author: shash
"""
import requests
from bs4 import BeautifulSoup
import time

import pandas as pd
df = pd.DataFrame(columns =['likes','about_post','date'])

url="https://www.instagram.com/virat.kohli/"
number_of_post=600

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

driver=webdriver.Chrome(executable_path="C:\\Program Files\\chromeDriverSelenium\\chromedriver.exe")
#driver.maximize_window()

driver.get(url)
driver.implicitly_wait(10)
print(driver.title)
#<div class="Nnq7C weEfm"><div class="v1Nh3 kIKUG  _bz0w"><a href="/p/CBctEw2lFwFmYLqzHRmje49-ikdEO-b_19yju40/" tabindex="0"><div class="eLAPa"><div class="KL4Bh"><img alt="Photo by Prabal Goel in Jaypee Institute of Information Technology with @sarthak_aagarwal, @siddharth9299, @garg.dhruv19, and @anandsingh5615. Image may contain: 9 people, people standing, text that says 'SEE REDMI NOTE7 NOT PRO Al DUAL CAMERA 2020/2/9 00:25'." class="FFVAD" decoding="auto" sizes="293.0078125px" srcset="https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c236.0.608.608a/s150x150/103422684_178499450284693_466448152927526586_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=101&amp;_nc_ohc=nc4IS2Q-xc0AX-wPxB4&amp;_nc_tp=16&amp;oh=c9ddc4c69ccdc70dab24fd220219c235&amp;oe=5FA49EDD 150w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c236.0.608.608a/s240x240/103422684_178499450284693_466448152927526586_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=101&amp;_nc_ohc=nc4IS2Q-xc0AX-wPxB4&amp;_nc_tp=16&amp;oh=7c9661630a80623f33873ee8b1869775&amp;oe=5FA2BF17 240w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c236.0.608.608a/s320x320/103422684_178499450284693_466448152927526586_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=101&amp;_nc_ohc=nc4IS2Q-xc0AX-wPxB4&amp;_nc_tp=16&amp;oh=6b61619bcad3ff680ba48263fd4a8313&amp;oe=5FA30D2D 320w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c236.0.608.608a/s480x480/103422684_178499450284693_466448152927526586_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=101&amp;_nc_ohc=nc4IS2Q-xc0AX-wPxB4&amp;_nc_tp=16&amp;oh=84b651fbe9f6e0f702e509e87e94cd9b&amp;oe=5FA5A177 480w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c236.0.608.608a/103422684_178499450284693_466448152927526586_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=101&amp;_nc_ohc=nc4IS2Q-xc0AX-wPxB4&amp;oh=f752d17d3ef393455e92ec5694c56a50&amp;oe=5FA21E29 640w" src="https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c236.0.608.608a/103422684_178499450284693_466448152927526586_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=101&amp;_nc_ohc=nc4IS2Q-xc0AX-wPxB4&amp;oh=f752d17d3ef393455e92ec5694c56a50&amp;oe=5FA21E29" style="object-fit: cover;"></div><div class="_9AhH0"></div></div></a></div><div class="v1Nh3 kIKUG  _bz0w"><a href="/p/B_FmQEyFy3emqlOhZcRJQybnJ3PQIFmcakb3XA0/" tabindex="0"><div class="eLAPa"><div class="KL4Bh"><img alt="Photo by Prabal Goel on April 17, 2020." class="FFVAD" decoding="auto" sizes="293.0078125px" srcset="https://instagram.fpat3-2.fna.fbcdn.net/v/t51.2885-15/e35/s150x150/93764360_1148445355497511_2045603743139205284_n.jpg?_nc_ht=instagram.fpat3-2.fna.fbcdn.net&amp;_nc_cat=107&amp;_nc_ohc=43PxUmne1G0AX8L3mQM&amp;_nc_tp=15&amp;oh=30ede8fb9845c4fa95e8e3fb5226077e&amp;oe=5FA30ECA 150w,https://instagram.fpat3-2.fna.fbcdn.net/v/t51.2885-15/e35/s240x240/93764360_1148445355497511_2045603743139205284_n.jpg?_nc_ht=instagram.fpat3-2.fna.fbcdn.net&amp;_nc_cat=107&amp;_nc_ohc=43PxUmne1G0AX8L3mQM&amp;_nc_tp=15&amp;oh=80e921a02929a4a263a3dac57f49b6f6&amp;oe=5FA411D0 240w,https://instagram.fpat3-2.fna.fbcdn.net/v/t51.2885-15/e35/s320x320/93764360_1148445355497511_2045603743139205284_n.jpg?_nc_ht=instagram.fpat3-2.fna.fbcdn.net&amp;_nc_cat=107&amp;_nc_ohc=43PxUmne1G0AX8L3mQM&amp;_nc_tp=15&amp;oh=34661319a68c3e6dadc1a3016374a30a&amp;oe=5FA47132 320w,https://instagram.fpat3-2.fna.fbcdn.net/v/t51.2885-15/e35/s480x480/93764360_1148445355497511_2045603743139205284_n.jpg?_nc_ht=instagram.fpat3-2.fna.fbcdn.net&amp;_nc_cat=107&amp;_nc_ohc=43PxUmne1G0AX8L3mQM&amp;_nc_tp=15&amp;oh=c0af3d43c17e51b72b028502a01bdb69&amp;oe=5FA4DA77 480w,https://instagram.fpat3-2.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s640x640/93764360_1148445355497511_2045603743139205284_n.jpg?_nc_ht=instagram.fpat3-2.fna.fbcdn.net&amp;_nc_cat=107&amp;_nc_ohc=43PxUmne1G0AX8L3mQM&amp;oh=186e6a18ce818fae7d74b715ca29f06b&amp;oe=5FA2394D 640w" src="https://instagram.fpat3-2.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s640x640/93764360_1148445355497511_2045603743139205284_n.jpg?_nc_ht=instagram.fpat3-2.fna.fbcdn.net&amp;_nc_cat=107&amp;_nc_ohc=43PxUmne1G0AX8L3mQM&amp;oh=186e6a18ce818fae7d74b715ca29f06b&amp;oe=5FA2394D" style="object-fit: cover;"></div><div class="_9AhH0"></div></div></a></div><div class="v1Nh3 kIKUG  _bz0w"><a href="/p/BtSqm9YFreKh2tgcxlLcATy9H1sOcA2mEpSuWE0/" tabindex="0"><div class="eLAPa"><div class="KL4Bh"><img alt="Photo by Prabal Goel in Red Carpet Party Lawn. Image may contain: 2 people." class="FFVAD" decoding="auto" sizes="293.0078125px" srcset="https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c60.0.959.959a/s150x150/50253178_263225194573251_3962242989193969584_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=103&amp;_nc_ohc=RuPt0iztwxMAX_YaGTS&amp;_nc_tp=16&amp;oh=ebdefba884fe7c7f50f208cf1ce28158&amp;oe=5FA37956 150w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c60.0.959.959a/s240x240/50253178_263225194573251_3962242989193969584_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=103&amp;_nc_ohc=RuPt0iztwxMAX_YaGTS&amp;_nc_tp=16&amp;oh=01dfc5a90dc1a0516781ef2dc52e6dc0&amp;oe=5FA49120 240w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c60.0.959.959a/s320x320/50253178_263225194573251_3962242989193969584_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=103&amp;_nc_ohc=RuPt0iztwxMAX_YaGTS&amp;_nc_tp=16&amp;oh=717ae9f87448ca52945a6d0add2ab835&amp;oe=5FA46EA6 320w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/e35/c60.0.959.959a/s480x480/50253178_263225194573251_3962242989193969584_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=103&amp;_nc_ohc=RuPt0iztwxMAX_YaGTS&amp;_nc_tp=16&amp;oh=3d9610f901d4ff61897f8a24180c215a&amp;oe=5FA3B780 480w,https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/c60.0.959.959a/s640x640/50253178_263225194573251_3962242989193969584_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=103&amp;_nc_ohc=RuPt0iztwxMAX_YaGTS&amp;oh=3d637ff3fa9ccec830342000012d35f3&amp;oe=5FA2B1E0 640w" src="https://instagram.fpat3-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/c60.0.959.959a/s640x640/50253178_263225194573251_3962242989193969584_n.jpg?_nc_ht=instagram.fpat3-1.fna.fbcdn.net&amp;_nc_cat=103&amp;_nc_ohc=RuPt0iztwxMAX_YaGTS&amp;oh=3d637ff3fa9ccec830342000012d35f3&amp;oe=5FA2B1E0" style="object-fit: cover;"></div><div class="_9AhH0"></div></div></a></div></div>
content = driver.find_element_by_xpath('/html/body/div[1]/section/main/div/div[2]/article/div[1]/div/div[1]/div[1]/a')
content.click()

#action = ActionChains(driver)
#action.move_

username=driver.find_element_by_xpath('/html/body/div[5]/div[2]/div/div[2]/div/div/div[1]/div/form/div[1]/div[1]/div/label/input')
#print(username)
username.send_keys("")

#driver.close()

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
#/html/body/div[1]/section/main/div/div[2]/article/div[1]/div/div[1]/div[1]/a
content = driver.find_element_by_xpath('/html/body/div[1]/section/main/div/div[2]/article/div[1]/div/div[1]/div[1]/a')
content.click()
first=0

likes=0
about_post='not known'
date='not know'

try:
    for i in range(1,number_of_post+1):
        print("photo number",i)
        if first==0:
            move=driver.find_element_by_xpath('/html/body/div[5]/div[1]/div/div/a')
            first=1
        else:
            move=driver.find_element_by_xpath('/html/body/div[5]/div[1]/div/div/a[2]')
        time.sleep(5)
        try:
            text=driver.find_element_by_xpath('/html/body/div[5]/div[2]/div/article/div[3]/div[1]/ul/div/li/div/div/div[2]')
            about_post=str(text.text)
            print(about_post)
        except:
            print("not known")
        try:
            count=driver.find_element_by_xpath('/html/body/div[5]/div[2]/div/article/div[3]/section[2]/div')
            likes=(count.text)
        except:
            likes=0
        try:
            date=driver.find_element_by_xpath('/html/body/div[5]/div[2]/div/article/div[3]/div[2]/a')
            date=(date.text)
        except:
            date="not known"
        to_append = [likes, about_post,date]
        df_length = len(df)
        df.loc[df_length] = to_append
       # print(df)
        move.click()
finally:
    df.to_csv('C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping\\about_post\\virat_kohli.csv',index=False)        
    driver.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
