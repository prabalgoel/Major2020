# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:49:27 2020

@author: shash
"""
#C:\Program Files\Tesseract-OCR
import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

roi=[[(3, 213), (65, 223), 'TEXT', 'impressions']]

img = cv2.imread('C:\\Users\\shash\\Desktop\\Data_Science\\Major2020\\images\\1st.jpeg') 
cv2.imshow('original image', img) 
"""
imgShow=img.copy()
imgMask=np.zeros_like(imgShow)

for x,r in enumerate(roi):
    cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][0]),(0,255,0),cv2.FILLED)
    imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
cv2.imshow('masked',imgShow)

"""
























