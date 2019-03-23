import numpy as np
import cv2
import os
from skvideo.io import VideoWriter
# if os.path.isfile('output.avi') and False:
#     quit()
#     os.system('mv output.avi output1.avi')
#     capo = cv2.VideoCapture('output1.avi')
#     fourccc=cv2.cv.CV_FOURCC('X','V','I','D')
#     o = cv2.VideoWriter('output.avi',fourccc, 20, (28,28))
#     while True:
#         ret, frame = capo.read()
#         if not ret:
#             break
#         o.write(frame)
#     capo.release()
# else:
#     pass
fourcc=cv2.cv.CV_FOURCC('X','V','I','D')
# fourcc = cv2.cv.CV_FOURCC(*'FMP4')
# o = cv2.VideoWriter('output.avi',fourcc, 60, (28,28),0)
o = VideoWriter("output.avi", frameSize=(28,28))
o.open()
assert(o.isOpened())
cap = cv2.VideoCapture('vid.avi')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

labels=open("digits.txt","a")
spos=0
try:
    pg=open("progress.txt","r")
    l=int(pg.readline())
    spos=l
except:
    "No Progress file"

kernel = np.ones((2,2),np.uint8)
i=1
s=None

p1=np.float32([[0,0],[0,1],[1,1]])
p2=np.float32([[-.1,0],[.1,1],[1.1,1]])
M=cv2.getAffineTransform(p1,p2)
ii=0
alive=True
more=True
imgs=[]
while alive:
    while(cap.isOpened() and len(imgs)<50 and more):
        if i%128==0:
            print i
        ret, frame = cap.read()
        if not ret:
            more=False
            break
        framec=frame[360:420,450:550]
        l = np.array([240,180,0])
        u = np.array([255,220,20])

        th=cv2.inRange(framec, l,u)
        th2=cv2.erode(th,kernel,iterations = 1)
        th3=cv2.dilate(th2,kernel,iterations = 1)
        # print th3.shape

        thf=cv2.bitwise_and(framec,framec,mask=th3)
        # s=th3
        rows,cols = th3.shape
        s = cv2.warpAffine(th3,M,(int(cols*1.3),rows))
        breaks=np.reshape(np.sum(s,axis=0),(1,-1))>5

        breaks=breaks
        going=False
        # ev=[]
        c=0
        # dd=[]
        for i in range(breaks.shape[1]):
            k=breaks[0,i]
            # print k
            if going and not k:
                # ev.append((c,i))
                rs=cv2.resize(s[:,c:i],(28,28))
                # dd.append(rs)
                imgs.append(rs)
                going=False
                # print("ending")
            elif not going and k:
                c=i
                # print("starting")
                going=True

        # print breaks
        dsep=np.tile(breaks*1.0,(60,1))
    while len(imgs)>20:
        ig=imgs.pop(0)
        if(ii>1000):
            alive=False
            break
        # if(spos>0):
        #     spos-=1
        #     print("skipping")
        #     ii+=1
        #     continue
        print ii
        iig=np.tile(np.reshape(ig,(28,28,1)),(1,1,3))
        # print iig.shape
        o.write(iig)
        # print "wr"
        # if(np.sum(ig)<5):
        #     key='0'
        #     print("Frame=%d Key=0auto"%(ii,))
        # else:
        #     ishow=np.hstack((ig,np.hstack(imgs[0:9])))
        #     cv2.imshow('frame',ishow)
        #     key=chr(cv2.waitKey(0))
        #     print("Frame=%d Key=%c"%(ii,key))
        ii+=1
        # if(key=='q'):
        #     alive=False
        #     break
        # labels.write("%c\n"%key)



    # cv2.imshow('frame',ishow)
    # cv2.imshow('frame2',s)
    # cv2.imshow('frame3',d2)
    # cv2.imshow('frame4',d3)
    
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #    break

cap.release()
o.release()
labels.close()
pgg=open("progress.txt","w")
pgg.write("%d\n"%ii)
pgg.close()
cv2.destroyAllWindows()