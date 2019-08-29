
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
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    """
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
     # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    """
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[:2]
    img_empty = np.zeros((height, width, 3))
    cv2.drawContours(img_empty, contours, -1, (0, 255, 0), 3)

    compare_images(img, img_empty)
