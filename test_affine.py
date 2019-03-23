import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("sc.png")
rows,cols,ch = img.shape
p1=np.float32([[0,0],[0,1],[1,1]])
p2=np.float32([[-.1,0],[.1,1],[1.1,1]])
M=cv2.getAffineTransform(p1,p2)
dst = cv2.warpAffine(img,M,(int(cols*1.3),rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()