'''
Created on 2017. 10. 19.

@author: danny
'''
import cv2 #opencv-python
import numpy as np
from PIL import Image
import pytesseract
def get_string(img_path):
    # Read image with opencv
    img = cv2.pyrDown(cv2.imread(img_path,cv2.IMREAD_UNCHANGED))

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th_plate  = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    cv2.imwrite('./Images/plate_th.jpg',th_plate)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((3, 3), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    er_plate = cv2.erode(th_plate, kernel, iterations=1)
    er_invplate = er_plate
    # Write image after removed noise
    cv2.imwrite("./Images/er_plate.jpg", er_invplate)

    result = pytesseract.image_to_string(Image.open("./Images/er_plate.jpg"),lang='eng')
    print(result.replace(" ",""))
    # Remove template file
    #os.remove(temp)
    return result.replace(" ","")

#print("OCR:",get_string("./images/try40.jpg"))
img_path = "./Images/test3.jpg"
im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)

#print(im)
print("OCR:",pytesseract.image_to_string(Image.open(img_path).convert("L"),lang='eng').replace(" ",""))
# print("OCR_T:",get_string(img_path))

cv2.imshow('BGR2GRAY',im)
k = cv2.waitKey(0)
if k == 27: # esc key
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('lenagray.png',img)
    cv2.destroyAllWindows()



# import cv2
# from matplotlib import pyplot as plt # as는 alias 적용시 사용
# 
# img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
# 
# b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
# img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
# 
# plt.imshow(img2)
# plt.xticks([]) # x축 눈금
# plt.yticks([]) # y축 눈금
# plt.show()
