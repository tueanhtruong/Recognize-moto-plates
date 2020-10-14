import cv2
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from joblib import load
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

clf = svm.SVC( C=100)
clf=load('D:/HK2N3/character2svm.joblib')
img = cv2.imread('./newdata/7.png',cv2.COLOR_BGR2GRAY)
img_pad = get_square(img,28)
img_pad = np.array(img_pad.reshape(1,-1)[0])/255

'''
print("\n==========\n",img_pad.reshape(1, -1)[0],"\n==========\n")
print("\n==========\n",img_pad.reshape(1, -1),"\n==========\n")
print(len(img_pad.reshape(1, -1)[0]))
'''
value = []
value.append (clf.predict(img_pad.reshape(1, -1))[0])
print("\n==========\n",value,type(value),"\n==========\n")
plt.imshow(img_pad.reshape(28, -1), cmap=plt.cm.get_cmap('gray') ,interpolation='nearest')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()