
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def compare_images(img1, img2):
    try:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    except:
        pass
    try:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    except:
        pass
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(img2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    files = glob.glob('Additional_test_data/small/*')
    img = cv2.imread(files[30])

    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(imgray, (5, 5), 0)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contours = sorted_contours[:10]

    height, width = img.shape[:2]
    img_empty = np.zeros((height, width, 3))
    cv2.drawContours(img_empty, largest_contours, -1, (0, 255, 0), 3)

    compare_images(img, img_empty)    