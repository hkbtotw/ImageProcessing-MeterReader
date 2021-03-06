from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
import os
import glob
import pandas as pd
import re
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from PIL import Image,  ImageEnhance

def LocateNumber(path):
    imageA = cv2.imread(path)
    hsv_img = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    #print(s, ' ==> ',s.shape)
    proj = np.sum(s,0)  # Compute Horizontal projection with 0 , Vertical with 1 as argument
    #print(' ==> ', np.mean(proj))

    nlen=len(proj)
    plist = proj.tolist()
    pindex=list(range(1,nlen+1))
    #print(' == ', proj, ' == ', type(proj), '  len : ',nlen, ' ; ',plist, ' ... ', type(plist), ' -- ', pindex)

    parray = np.asarray(pindex)
    dataarray = np.asarray(plist)

    # Fit data with polynomail order 30th
    zp = np.polyfit(parray, dataarray, 30) 
    p = np.poly1d(zp)
    pin=p(pindex)    
    _ = plt.plot(pindex, dataarray, '.', pindex, pin, '-')

    peaks, _ = find_peaks(-pin, prominence=1.5)
    #print(' peak x :', peaks)

    plt.plot(proj)
    plt.plot(peaks, pin[peaks], "x")
    plt.show()

    return peaks

def LocateVerticalPoint(path):
    imageA = cv2.imread(path)
    hsv_img = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    #print(s, ' ==> ',s.shape)
    proj = np.sum(s,1)  # Compute Horizontal projection with 0 , Vertical with 1 as argument
    #print(' :: ',proj)
    #print(' ==> ', np.mean(proj) )

    plen=len(proj)
    pstart=proj[0]
    pend=proj[plen-1]
    #print(' > ', plen, ' ==> ', pstart, ', ', pend)
    pStartRange=[0.8*pstart, 1.2*pstart]
    pEndRange=[0.8*pend, 1.2*pend]

    #print(' >> ', pStartRange, ' == ', pEndRange)
    pStartCut=0
    pEndCut=plen-1
    for n in range(0,plen):
        if(proj[n]<pStartRange[0] or proj[n]>pStartRange[1]):
            pStartCut=n
            #print(' pStartCut >>> ', pStartCut)
            break
    for n in range(plen-1,0,-1):
        #print(' >>>> ', n)
        if(proj[n]<pEndRange[0] or proj[n]>pEndRange[1]):
            pEndCut=n
            #print(' pEndCut >>> ', pEndCut)
            break
    #plt.plot(proj)
    #plt.plot(peaks, pin[peaks], "x")
    #plt.show()
    #print(' >>> ', pStartCut, ' :: ', pEndCut)
    return pStartCut, pEndCut

image_path=r"C:/Users/70018928/Documents/Project2020/TruckOdometer/20200203/Test_SSM_1/image2/"
path=image_path+"*.jpg"
files = []
for file in glob.glob(path):
    files.append(file)

for n in files:
    cropPoint=LocateNumber(n)
    cropList=list(cropPoint)
    #print(n, ' CP :', cropList)

    image = Image.open(n)
    im_width, im_height = image.size
    #im_height, im_width, channels = image.shape
    #print(' image : ',type(image),' ==> ',im_width,':',im_height)

    Top, Bottom=LocateVerticalPoint(n)

    ilen=len(cropList)
    for cp in range(0,ilen-1):
        x1=cropList[cp]
        x2=cropList[cp+1]
        y1=Top #0
        y2=Bottom #im_height
        #print(' : ',x1,' : ',x2,' : ', y1,' : ', y2)
        area=(x1,y1,x2,y2)
        cropped_img=image.crop(area)
        cropped_img.show()
        plt.show()









