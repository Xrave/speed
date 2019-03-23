import numpy as np
import cv2
import os
# from skvideo.io import VideoWriter
import theano
import theano.tensor as T
import lasagne
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

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

input_var = T.tensor4('input_var')
network=build_cnn(input_var=input_var)
# And load them again later on like this:
with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
pred=T.argmax(test_prediction, axis=1)
val_fn = theano.function([input_var], [pred])

# fourcc=cv2.cv.CV_FOURCC('X','V','I','D')
# # fourcc = cv2.cv.CV_FOURCC(*'FMP4')
# # o = cv2.VideoWriter('output.avi',fourcc, 60, (28,28),0)
# o = VideoWriter("output.avi", frameSize=(28,28))
# o.open()
# assert(o.isOpened())
cap = cv2.VideoCapture('vid.avi')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

# labels=open("digits.txt","a")
# spos=0
# try:
#     pg=open("progress.txt","r")
#     l=int(pg.readline())
#     spos=l
# except:
#     "No Progress file"

kernel = np.ones((2,2),np.uint8)
i=1
s=None

p1=np.float32([[0,0],[0,1],[1,1]])
p2=np.float32([[-.1,0],[.1,1],[1.1,1]])
M=cv2.getAffineTransform(p1,p2)
ii=0
alive=True
more=True

# while alive:
while(cap.isOpened() and  more):
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
    imgs=[]
    # print(breaks.shape[1])
    for i in range(breaks.shape[1]):
        k=breaks[0,i]
        # print k
        if going and not k:
            # ev.append((c,i))
            rs=cv2.resize(s[:,c:i],(28,28))
            # dd[i,0,:,:]=rs
            imgs.append(rs)
            going=False
            # print("ending")
        elif not going and k:
            c=i
            # print("starting")
            going=True
    # print len(imgs)
    dd=np.empty((len(imgs),1,28,28),dtype='float32')
    for i in range(len(imgs)):
        dd[i,0,:,:]=imgs[i]
    # guess= np.argmax(lasagne.layers.get_output(network, dd,deterministic=True).eval(),axis=1)
    ga=val_fn(dd)[0]
    guess=0
    for gae in ga:
        guess=guess*10+gae
    print guess
    cv2.imshow('frame',s)
    cv2.waitKey(1)
    # print breaks
    # dsep=np.tile(breaks*1.0,(60,1))
    # while len(imgs)>20:
    #     ig=imgs.pop(0)
    #     if(ii>1000):
    #         alive=False
    #         break
    #     # if(spos>0):
    #     #     spos-=1
    #     #     print("skipping")
    #     #     ii+=1
    #     #     continue
    #     print ii
    #     iig=np.tile(np.reshape(ig,(28,28,1)),(1,1,3))
    #     # print iig.shape
    #     o.write(iig)
    #     # print "wr"
    #     # if(np.sum(ig)<5):
    #     #     key='0'
    #     #     print("Frame=%d Key=0auto"%(ii,))
    #     # else:
    #     #     ishow=np.hstack((ig,np.hstack(imgs[0:9])))
    #     #     cv2.imshow('frame',ishow)
    #     #     key=chr(cv2.waitKey(0))
    #     #     print("Frame=%d Key=%c"%(ii,key))
    #     ii+=1
    #     # if(key=='q'):
    #     #     alive=False
    #     #     break
    #     # labels.write("%c\n"%key)



    # cv2.imshow('frame',ishow)
    # cv2.imshow('frame2',s)
    # cv2.imshow('frame3',d2)
    # cv2.imshow('frame4',d3)
    
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #    break

cap.release()
# o.release()
# labels.close()
# pgg=open("progress.txt","w")
# pgg.write("%d\n"%ii)
# pgg.close()
cv2.destroyAllWindows()