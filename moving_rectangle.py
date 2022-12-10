import cv2 as cv
import numpy as np
from time import sleep

mat = np.zeros((480,640), dtype=np.uint8)
mat[200:280,280:360] = 255

for _ in range(1000):
    mat = np.roll(mat, 10, (0,1))
    cv.imshow("mat", mat)
    cv.waitKey(300)