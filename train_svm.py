import cv2
import numpy as np
import random
from sklearn import svm
from joblib import dump
import os

def get_square(image,square_size):

    (height,width)=image.shape[:2]
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4
    mask = np.zeros((differ,differ), dtype="uint8")
    mask+=255
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)
    return mask


path= 'D:/HK2N3/data2'
list_label = os.listdir(path)
train_label=[];         # nhan dung de train
train_data=[];          # data dung de train
for a in list_label:    # doc anh va train tu file du lieu
    if(list_label.index(a)<=32):
        sub_list = os.listdir(path+"/"+a)
        for b in sub_list:
            img = cv2.imread(path+"/"+a+"/"+b)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = get_square(img,28)
            train_data.append(img.reshape(1,-1)[0])
            train_label.append(int(a))

print(len(train_label),"---",len(train_data))   # so anh da train

# dao thu tu cac anh va label tuong ung
shuffle_order = list(range(len(train_label)))
random.shuffle(shuffle_order)
train_data = np.array(train_data)
train_label = np.array(train_label)

train_data = train_data[shuffle_order]
train_label = train_label[shuffle_order]


model=svm.SVC(C=0.001,kernel="poly",gamma=10)        # khai bao module
train_data = np.array(train_data)/255   # chuan hoa anh ve 0 -> 1
model.fit(train_data,train_label)       # train anh
dump(model,'D:/HK2N3/character2svm.joblib')     # luu module



