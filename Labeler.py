import cv2
import os
import glob
from os import path
import random
mypath='./Images/road'
testpath='./test'
newpath = './train'
label='road.'
if not os.path.exists(newpath):
    os.makedirs(newpath)

def get_traindata():
    img_num=1
    for n in os.listdir(mypath):
        path=os.path.join(mypath,n)
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img= cv2.resize(img , (224,224))
        cv2.imwrite('./train/'+label+'new'+str(img_num)+'.jpg',img)
        img_num+=1
get_traindata()
