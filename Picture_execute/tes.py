import cv2 as cv
import numpy as np

# 1.导入图片
img_org = cv.imread('./eight.jpg', cv.IMREAD_GRAYSCALE)

# 2.二值化处理
ret, img_bin = cv.threshold(img_org, 128, 255, cv.THRESH_BINARY_INV)

# 3.细化前处理
kernel = np.ones((3, 3), np.uint8)
img_bin = cv.erode(img_bin, kernel, iterations=1)
img_bin = cv.dilate(img_bin, kernel, iterations=1)

# 4.细化处理
img_thinning = cv.ximgproc.thinning(img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

# 5.显示结果
cv.imshow('img_org', img_org)
cv.imshow('img_bin', img_bin)
cv.imshow('img_thinning', img_thinning)

cv.waitKey()
cv.destroyAllWindows()

