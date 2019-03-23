import numpy as np
import cv2
import theano
import theano.tensor as T
import lasagne

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


cap = cv2.VideoCapture('vid.avi')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc=cv2.cv.CV_FOURCC('X','V','I','D')
o = cv2.VideoWriter('output.avi',fourcc, 20, (100,60))

kernel = np.ones((2,2),np.uint8)
i=1
# print o.isOpened()
s=None
network=build_cnn()
# And load them again later on like this:
with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
pred=T.argmax(test_prediction, axis=1)

# quit()

while(cap.isOpened()):
    i+=1
    if i%128==0:
        print i
    ret, frame = cap.read()
    if not ret:
        break
    framec=frame[360:420,450:550]
    l = np.array([240,180,0])
    u = np.array([255,220,20])

    th=cv2.inRange(framec, l,u)
    th2=cv2.erode(th,kernel,iterations = 1)
    th3=cv2.dilate(th2,kernel,iterations = 1)
    # print th3.shape
    thf=cv2.bitwise_and(framec,framec,mask=th3)
    if s==None:
        s=th3
    else:
        s=th3
    o.write(framec)
    # if i==100:
    #     break
    # if(i==100):
    #     cv2.imwrite("f100.jpg",th3)
    dm=np.empty((4,1,28,28),dtype='float32')

    d0=cv2.resize(s[:,74:96],(28,28))
    d1=cv2.resize(s[:,51:74],(28,28))
    d2=cv2.resize(s[:,28:51],(28,28))
    d3=cv2.resize(s[:,5:28],(28,28))
    dm[0,:,:,:]=d0
    dm[1,:,:,:]=d1
    dm[2,:,:,:]=d2
    dm[3,:,:,:]=d3
    guess= np.argmax(lasagne.layers.get_output(network, dm,deterministic=True).eval(),axis=1)
    num=0
    for i in range(4):
        if(np.sum(dm[i,:,:,:])<20):
            guess[i]=0
        num+=guess[i]*(10**i)
    print num
    # cv2.imshow('frame',d0)
    # cv2.imshow('frame2',d1)
    # cv2.imshow('frame3',d2)
    # cv2.imshow('frame4',d3)
    cv2.imshow('frame',s)
    if cv2.waitKey(500) & 0xFF == ord('q'):
       break

cap.release()
o.release()
cv2.destroyAllWindows()