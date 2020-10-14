import cv2
import numpy as np
from sklearn import svm
from joblib import load

def get_square(image,square_size):  # ham dung de resize anh ve kich thuoc 28x28

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

def filpoint(n):
    if n>20 and n<253:
        return n
    else:
        if n>=253:
            return n-70
        else:
            return 0

save=0
src_path = "D:/HK2N3/Bike_back/"               # duong dan file anh,
img = cv2.imread(src_path+"53.jpg")     # doc anh
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # Chuyen sang anh xam
gray=cv2.equalizeHist(gray)             # can bang dai mau anh
blur = cv2.blur(gray,(5,5))             # lam min anh
i=0

for m in gray:
    a = map(filpoint,m)
    gray[i]= (list(a))
    i+=1

cv2.imshow("IMG",gray)
thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,5)
cv2.imshow("THRES_0",thres)
# nhi phan anh dung nguong dong
ret,thress = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)  # nhi phan anh nguong 200 (tuy chon)
cv2.imshow("THRES_1",thress)
thres1 = cv2.bitwise_and(thres,thress)  # tong hop 2 anh nhi phan tren => ta duoc 1 anh it chi tiet nen hon

cv2.imshow("THRES_2",thres1) # show anh nhi phan


element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
thres=cv2.erode(thres1,element)     # co anh, lam ro cac vien den
contour,hier=cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# tim duong bao quanh cac vung trang trong anh nhi phan
contour=sorted(contour,key=cv2.contourArea,reverse= True)[:10]
# sap xep lay 10 vung dien tich lon nhat

rect=None
x=0
y=0
w=0
h=0
rong = img.shape[1]     # do rong cua anh
cao = img.shape[0]      # do cao cua anh
screenCnt = None
re= None
recolor=[]
sub_char = []
list_rect = []
count=len(sub_char)

for c in contour:
    rect = cv2.boundingRect(c) # lay duong bao la hinh chu nhat quanh cac vung trang da chon
    x=rect[0]   # toa do x
    y=rect[1]   # toa do y
    w=rect[2]   # chieu rong
    h=rect[3]   # chieu cao


    if w/h>1.0 and w/h<2.0 and rong/w>1.25 and cao/h>1.25 and cv2.contourArea(c)>9000:    # xet dieu kien
        re1 = thres1[y:y+h,x:x+w].copy()
        # thoa man ta tiep tuc tim cac chu so trong do
        sub_contour,subhier=cv2.findContours(re1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(sub_contour) < 8:
            continue    # so chu so nho hon 8 thi loai
        for cc in sub_contour:  # tuong tu dua vao ti le tim cac chu so trong hinh
            sub_rect = cv2.boundingRect(cc)
            sx=sub_rect[0]
            sy=sub_rect[1]
            sw=sub_rect[2]
            sh=sub_rect[3]
            if sh<h/2.0 and sw<w/6.0 and sh>h/4.0 and sw>20 and sh>=60 and sx>2:
                cj = re1[sy:sy+sh,sx:sx+sw]
                ratio = cv2.countNonZero(cj)/(cj.shape[0]*cj.shape[1])
                if ratio >0.2 and ratio<0.7:
                    sub_char.append(cj)
                    list_rect.append(sub_rect)
        if count!=len(sub_char):
            sub_char=sub_char[count:len(sub_char)]
            list_rect=list_rect[count:len(list_rect)]
            re=re1.copy()
            recolor = img[y:y+h,x:x+w].copy()
            count=len(sub_char)



#sap xep cac chu cai
num = len(list_rect)
mean_high=0
index = list(range(num))
for i in index[:num-1]:
    mean_high= mean_high+list_rect[i][1]
    for j in index[i+1:]:
        if list_rect[i][0]>list_rect[j][0]:
            tam1 = list_rect[i]
            list_rect[i] = list_rect[j]
            list_rect[j] = tam1
            tam2 = sub_char[i]
            sub_char[i] = sub_char[j]
            sub_char[j] = tam2
if num!=0:
    mean_high=mean_high/num
print(num, index,mean_high)
sub_char2=[]
list_rect2=[]
for i in index:
    if list_rect[i][1]<mean_high:
        sub_char2.append(sub_char[i])
        list_rect2.append(list_rect[i])
for i in index:
    if list_rect[i][1]>mean_high:
        sub_char2.append(sub_char[i])
        list_rect2.append(list_rect[i])

#lay cac chu cai
num = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
       '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']
character=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
           'K','L','M','N','P','S','T','U','V','X','Y','Z','W','R','Q']
i=0
clf = svm.SVC( C=100)
clf=load('D:/HK2N3/character2svm.joblib')
value = []
index = []
print(clf.kernel)
if len(sub_char) != 0:
    for ccc in sub_char2:
        print(list_rect2[i])
        #cv2.imshow(num[i+save],ccc)
        cv2.imwrite("./newdata/"+num[i+save]+".png",ccc)    # luu anh cac chu so vao muc newdata
        img_pad = get_square(ccc, 28)   # resize ve 28x28
        img_pad = np.array(img_pad.reshape(1, -1)[0]) / 255     # chuan hoa ve dang 0 -> 1
        index.append(clf.predict(img_pad.reshape(1, -1))[0])
        value.append(character[index[i]])     # du doan ket qua
        i+=1
cv2.drawContours(img,contour,-1,(255,0,0),2)
if len(recolor) != 0:
    cv2.imshow("RESULT", recolor)
    print("INDEX:   ", index )
    print("RESULT:  ", value)
else:
    print("CAN'T RECOGNIZE")
#print(type(recolor))







cv2.waitKey(0)
cv2.destroyAllWindows()