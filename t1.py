#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    # print(dp1.shape)
    # print(dp1.shape[0])
    # print(len(dp1))
    # print(dp1.shape[1])
    # print(len(dp1[0]))
    # print(dp1[0][1])

    # Creating Borders around Image 2

    img2 = cv2.copyMakeBorder(img2, 150, 150, 150, 150, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Using SIFT to extract keypoint and descriptor features on both images
    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1, dp1 = sift.detectAndCompute(gray1, None)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, dp2 = sift.detectAndCompute(gray2, None)

    # finding sum of square differences between two images
    ssd = np.sqrt(np.square(dp1[:, None] - dp2).sum(axis=2))


    # finding minimum ssd and retaing its index1- row= keypoint of image 1 and index2 - column = keypoint of image 2. argmin giving index of the minimum. Shows min value's row - keypoints of image 1  and column  keypoints of image 2.
    min_idx = {idx[0]: m_idx for idx, m_idx in np.ndenumerate(np.argmin(ssd, axis=1))}

    # finding minimum ssd and retaing its index1- row= keypoint of image 1 and index2 - column = keypoint of image 2. Shows minimum value with row - kp of Image 1
    min_ssd = {idx[0]: m_idx for idx, m_idx in np.ndenumerate(np.min(ssd, axis=1))}

    # sorting SSDS according to there values
    lowest_ssds = dict(sorted(min_ssd.items(), key=lambda x: x[1]))
    # thresholding points
    lowest_img1_pts = list(lowest_ssds.keys())[:30]
    # Retracing indexes(row, column) of sorted thresholding points and saving it as best points
    best_pts = {k: v for k, v in min_idx.items() if k in lowest_img1_pts}

    #  finding corresponding KeyPoints in best points
    src_img1 = np.float32([kp1[k].pt for k, v in best_pts.items()])
    dest_img2 = np.float32([kp2[v].pt for k, v in best_pts.items()])

    # finding Homography matrix = val
    val, mask = cv2.findHomography(src_img1, dest_img2, cv2.RANSAC, 5.0)

    # warping of image1 of size : height of image2 and width of image 1 + image 2
    dst = cv2.warpPerspective(img1, val, ((img1.shape[1] + img2.shape[1]), img2.shape[0]))

    # Resizing the size of clippedimage to Image 2
    clippedimg = dst[0:img2.shape[0], 0:img2.shape[1]]


    h, w, _ = clippedimg.shape
    # creating new image of same size as clipped image and image 2
    new = np.zeros((h, w, _))
    for i in range(h):
        for j in range(w):
            # Eliminating the foreground based on pixel values

            if abs(np.sum(clippedimg[i][j])) == abs(np.sum(img2[i][j])):
                new[i][j] = img2[i][j]
            elif abs(np.sum((clippedimg[i][j]))) < abs(np.sum(img2[i][j])):
                new[i][j] = img2[i][j]
            elif abs(np.sum(clippedimg[i][j])) > abs(np.sum(img2[i][j])):
                new[i][j] = clippedimg[i][j]
    # cv2.imshow('NEWIMAGE',new)
    # cv2.waitKey(0)
    cv2.imwrite(savepath,new)
    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)






    # References:
    # https: // docs.opencv.org / 3.4 / dc / dc3 / tutorial_py_matcher.html

