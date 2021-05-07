# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:58:36 2020

@author: shash
"""
from pathlib import Path
import os
import cv2
import pytesseract
import numpy as np

rootdir=('C:\\Users\\shash\\Desktop\\Data_Science\\Major2020\\images')

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
scale =0.5

for subdir, dirs, files in os.walk(rootdir):
   for file in files:
       circles=[]
       counter=0
       counter2=0
       points1=[]
       points2=[]
       myPoints=[]
       myColor=[]

       # read image
       counter=0
       img = cv2.imread(subdir+"\\6.jpeg")
       img=cv2.resize(img,(0,0),None,scale,scale)
       #cv2.imshow('original image', img)
       def mouse_click(event,x,y,flags,param):
           global circles, counter, counter2, points1, points2, myPoints, myColor
           if event==cv2.EVENT_LBUTTONDOWN:
               if counter==0:
                   points1=int(x),int(y)
                   counter+=1
                   myColor=(random.randint(0,2)*200, random.randint(0,2)*200, random.randint(0,2)*200)
                   print(x,y,myColor)
               elif counter==1:
                   points2=int(x),int(y)
                   counter=0
                   print('out')
                   type='TEXT'
                   name='Date'
                   myPoints.append([points1,points2,type,name])
               circles.append([x,y,myColor])
               counter2+=1
               for x,y,color in circles:
                   cv2.circle(img,(x,y),3,color,cv2.FILLED)
               cv2.imshow("original image",img)


       #cv2.setMouseCallback('original image', mouse_click)

       cv2.waitKey(0)

       # close all the opened windows.
       print(myPoints)
       cv2.destroyAllWindows()



       roi=[

       [(123, 76), (183, 89), 'TEXT', 'Date'],
       [(212, 230), (262, 250), 'TEXT', 'Impressions'],
       [(224, 283), (262, 306), 'TEXT', 'Account Activity'],
       [(229, 321), (262, 336), 'TEXT', 'Profile Vists'],
       [(236, 368), (264, 383), 'TEXT', 'Website Tags']
            ]

       img = cv2.imread(subdir+"\\1.jpeg")
       print(subdir+"\\1.jpeg")
       print(type(img))
       img=cv2.resize(img,(0,0),None,scale,scale)

       print(img.shape)
       imgShow=img.copy()
       print(imgShow.shape)
       imgMask=np.zeros_like(imgShow)

       myData=[]

       img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       for x,r in enumerate(roi):
           cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,0,255),cv2.FILLED)
           imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
           imgCrop=img[r[0][1]:r[1][1],r[0][0]:r[1][0]]
           print(r[0][1],r[1][1],r[0][0],r[1][0])
           myData.append(pytesseract.image_to_string(imgCrop))
        #   cv2.imshow('croped'+str(x),imgCrop)
       #cv2.imshow('masked',imgShow)

       cv2.waitKey(0)
       cv2.destroyAllWindows()
       print(myData)





       scale=0.25
       roi=[
       [(221, 56), (264, 80), 'TEXT', 'Post Interactions'],
       [(224, 109), (263, 125), 'TEXT', 'Post Likes'],
       [(239, 141), (263, 156), 'TEXT', 'Post Comments'],
       [(242, 203), (263, 222), 'TEXT', 'Post Shares'],
       [(221, 241), (266, 261), 'TEXT', 'Story Interactions'],
       [(226, 291), (263, 306), 'TEXT', 'Story Replies'],
       [(231, 321), (265, 337), 'TEXT', 'Story Shares'],
       [(228, 356), (266, 376), 'TEXT', 'IGTV Interactions'],
       [(232, 407), (262, 424), 'TEXT', 'IGTV Likes']
            ]

       img = cv2.imread(subdir+"\\2.jpeg")
       img=cv2.resize(img,(0,0),None,scale,scale)

       print(img.shape)
       imgShow=img.copy()
       print(imgShow.shape)
       imgMask=np.zeros_like(imgShow)

       #myData=[]

       img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       for x,r in enumerate(roi):
           cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,0,255),cv2.FILLED)
           imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
           imgCrop=img[r[0][1]:r[1][1],r[0][0]:r[1][0]]
           print(r[0][1],r[1][1],r[0][0],r[1][0])
           myData.append(pytesseract.image_to_string(imgCrop))
          # cv2.imshow('croped'+str(x),imgCrop)
       #cv2.imshow('masked',imgShow)

       cv2.waitKey(0)
       cv2.destroyAllWindows()
       print(myData)




       scale=0.5
       roi=[[(461, 189), (503, 224), 'TEXT', '13-17'],
       [(469, 244), (497, 267), 'TEXT', '18-24'],
       [(469, 287), (496, 317), 'TEXT', '25-34'],
       [(470, 336), (497, 368), 'TEXT', '35-44'],
       [(462, 386), (503, 417), 'TEXT', '45-54'],
       [(462, 428), (505, 468), 'TEXT', '55-64'],
       [(467, 475), (505, 521), 'TEXT', '65+']
            ]

       img = cv2.imread(subdir+"\\3.jpeg")
       img=cv2.resize(img,(0,0),None,scale,scale)

       print(img.shape)
       imgShow=img.copy()
       print(imgShow.shape)
       imgMask=np.zeros_like(imgShow)

       #myData=[]

       img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       for x,r in enumerate(roi):
           cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,0,255),cv2.FILLED)
           imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
           imgCrop=img[r[0][1]:r[1][1],r[0][0]:r[1][0]]
           print(r[0][1],r[1][1],r[0][0],r[1][0])
           myData.append(pytesseract.image_to_string(imgCrop))
          # cv2.imshow('croped'+str(x),imgCrop)
       #cv2.imshow('masked',imgShow)

       cv2.waitKey(0)
       cv2.destroyAllWindows()
       print(myData)






       roi=[[(461, 189), (503, 224), 'TEXT', '13-17'],
       [(469, 244), (497, 267), 'TEXT', '18-24'],
       [(469, 287), (496, 317), 'TEXT', '25-34'],
       [(470, 336), (497, 368), 'TEXT', '35-44'],
       [(462, 386), (503, 417), 'TEXT', '45-54'],
       [(462, 428), (505, 468), 'TEXT', '55-64'],
       [(467, 475), (505, 521), 'TEXT', '65+']
            ]

       img = cv2.imread(subdir+"\\4.jpeg")
       img=cv2.resize(img,(0,0),None,scale,scale)

       print(img.shape)
       imgShow=img.copy()
       print(imgShow.shape)
       imgMask=np.zeros_like(imgShow)

       #myData=[]

       img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       for x,r in enumerate(roi):
           cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,0,255),cv2.FILLED)
           imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
           imgCrop=img[r[0][1]:r[1][1],r[0][0]:r[1][0]]
           print(r[0][1],r[1][1],r[0][0],r[1][0])
           myData.append(pytesseract.image_to_string(imgCrop))
        #   cv2.imshow('croped'+str(x),imgCrop)
       #cv2.imshow('masked',imgShow)

       cv2.waitKey(0)
       cv2.destroyAllWindows()
       print(myData)




       roi=[[(461, 189), (503, 224), 'TEXT', '13-17'],
       [(469, 244), (497, 267), 'TEXT', '18-24'],
       [(469, 287), (496, 317), 'TEXT', '25-34'],
       [(470, 336), (497, 368), 'TEXT', '35-44'],
       [(462, 386), (503, 417), 'TEXT', '45-54'],
       [(462, 428), (505, 468), 'TEXT', '55-64'],
       [(467, 475), (505, 521), 'TEXT', '65+']
            ]

       img = cv2.imread(subdir+"\\5.jpeg")
       img=cv2.resize(img,(0,0),None,scale,scale)

       imgShow=img.copy()
       imgMask=np.zeros_like(imgShow)

       #myData=[]

       img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       for x,r in enumerate(roi):
           cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,0,255),cv2.FILLED)
           imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
           imgCrop=img[r[0][1]:r[1][1],r[0][0]:r[1][0]]
           print(r[0][1],r[1][1],r[0][0],r[1][0])
           myData.append(pytesseract.image_to_string(imgCrop))
         #  cv2.imshow('croped'+str(x),imgCrop)
       #cv2.imshow('masked',imgShow)

       cv2.waitKey(0)
       cv2.destroyAllWindows()
       print(myData)










       roi=[
       [(172, 112), (255, 144), 'TEXT', 'Followers'],
       [(469, 351), (534, 373), 'TEXT', 'new_follow'],
       [(466, 377), (527, 414), 'TEXT', 'unfollow']

       ]

       img = cv2.imread(subdir+"\\6.jpeg")
       img=cv2.resize(img,(0,0),None,scale,scale)

       print(img.shape)
       imgShow=img.copy()
       print(imgShow.shape)
       imgMask=np.zeros_like(imgShow)

       #myData=[]

       img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       for x,r in enumerate(roi):
           cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,0,255),cv2.FILLED)
           imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
           imgCrop=img[r[0][1]:r[1][1],r[0][0]:r[1][0]]
           print(r[0][1],r[1][1],r[0][0],r[1][0])
           myData.append(pytesseract.image_to_string(imgCrop))
        #   cv2.imshow('croped'+str(x),imgCrop)
       #cv2.imshow('masked',imgShow)

       cv2.waitKey(0)
       cv2.destroyAllWindows()
       print(myData)





       import re
       with open('data.csv','a+') as f:
           for data in myData:
               data=re.sub("<'\n\x0c'>",'',data)
               data=data[:-1].rstrip("\n")
               print(data)
               #try:
               s=list(data)
               # print(s)
               for i in range(0,len(s)):
                   if ((ord(s[i])>=48 and ord(s[i])<=57) or (ord(s[i])>=65 and ord(s[i])<=90) or (ord(s[i])>=97 and ord(s[i])<=122) or s[i]=='-' or s[i]=='.'):
                       s[i]=s[i]
                   else:
                       s[i]=''
               new_data="".join(s)
               print(new_data)
               #except:
                   #new_data="unknown"
               print(new_data)
               f.write((str(new_data)+','))
           f.write('\n')       
       break
       





       