import cv2
num=0
for i in range(999):
    path='./real_raw/'+str(i)+'.jpg'
    image=cv2.imread(path)
    try:
        image=cv2.resize(image,(128,128),interpolation = cv2.INTER_AREA)
        cv2.imwrite('./real/'+str(num)+'.jpg',image)
        num+=1
    except:
        print(i)
        pass