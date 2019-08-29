
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def compare_images(images):

    plt.figure(figsize=(10,5))
    columns = len(images)

    for index, img in enumerate(images, 1):

        plt.subplot(1 ,columns, index)
        plt.axis('off')
        plt.imshow(img)

    plt.tight_layout()
    plt.show()

def binarize(gray_img):
    _, thresh = cv2.threshold(gray_img, 127, 255, 0)

    return thresh

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contours = sorted_contours[:10]

    return largest_contours

if __name__ == '__main__':
    files = glob.glob('Additional_test_data/small/*')
    img = cv2.imread(files[30])

    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(imgray, (5, 5), 0)

    binary_img = binarize(blurred_image)
    contours = find_contours(binary_img)

    height, width = img.shape[:2]
    img_empty = np.zeros((height, width, 3))
    cv2.drawContours(img_empty, contours, -1, (0, 255, 0), 3)

    compare_images([img, img_empty, binary_img])