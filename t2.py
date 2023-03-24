# 1. Only add your code inside the function (including newly improted packages).
#  You can design a new function and call the new function in the given functions.
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json



def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    # if imgmark == 't3':
    #     imgs = [cv2.resize(i, None, fx=200, fy=600, interpolation=cv2.INTER_AREA) for i in imgs]

    def stitching(img1, img_border):
        # Creating Borders around Image 2 to avoid cropping of image
        img2 = cv2.copyMakeBorder(img_border, 750, 750, 750, 750, cv2.BORDER_CONSTANT, (0, 0, 0))

        # Using SIFT to extract  2000 keypoint and descriptor features on both images
        sift = cv2.SIFT_create(2000)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        kp1, dp1 = sift.detectAndCompute(gray1, None)

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp2, dp2 = sift.detectAndCompute(gray2, None)

        # finding sum of square differences between two images
        d = dp1[:, None] - dp2
        sd = d*d
        ssd1 = sd.sum(axis = 2)
        ssd = np.sqrt(ssd1)
        # ssd = np.sqrt(np.square(dp1[:, None] - dp2).sum(axis=2))

        # finding minimum ssd and retaing its index1- row= keypoint of image 1 and index2 - column = keypoint of
        # image 2. argmin giving index of the minimum. Shows min value's row - keypoints of image 1 and column  keypoints of image 2
        min_idx = {idx[0]: m_idx for idx, m_idx in np.ndenumerate(np.argmin(ssd, axis=1))}

        # finding minimum ssd and retaing its index1- row= keypoint of image 1 and index2 - column = keypoint of
        # image 2. Shows minimum value with row - kp of Image 1
        min_ssd = {idx[0]: m_idx for idx, m_idx in np.ndenumerate(np.min(ssd, axis=1))}

        # sorting SSDS according to there values
        lowest_ssds = dict(sorted(min_ssd.items(), key=lambda row: row[1]))

        # thresholding to minimum 30 points
        lowest_img1_pts = list(lowest_ssds.keys())[:30]

        # Retracing indexes(row, column) of sorted back, thresholding points and saving it as best points
        best_pts = {key: val for key, val in min_idx.items() if key in lowest_img1_pts}

        # counting number of best minimum ssds'.
        count = 0
        for key, val in min_ssd.items():
            # keeping ssds below 100
            if val < 100.0:
                count += 1

        #  finding corresponding KeyPoints in best points
        src_img1 = np.float32([kp1[key].pt for key, val in best_pts.items()])
        dest_img2 = np.float32([kp2[val].pt for key, val in best_pts.items()])

        # finding Homography matrix = val
        val, mask = cv2.findHomography(src_img1, dest_img2, cv2.RANSAC, 5.0)

        # warping of image1 of size : height of image2 and width of image 1 + image 2
        dst = cv2.warpPerspective(img1, val, ((img1.shape[1] + img2.shape[1]), img2.shape[0]))

        # Resizing the size of clipped image to Image 2
        clipped_img = dst[0:img2.shape[0], 0:img2.shape[1]]

        # stitching warp image and image with border
        img_stitch = cv2.bitwise_or(img2, clipped_img)

        return count, img_stitch

    # Construct One Hot Array
    # For stitching each image with all others which contains 1 in the matrix
    # e.g. if 4 images - 4 x 4 Matrix
    matrix = []
    for k in imgs:
        temp = []
        for v in imgs:
            temp.append(stitching(k, v)[0])
        matrix.append(temp)
    one_hot_array = np.array(matrix)

    # Approximating Counts of the minimum distances as threshold - 160
    oa = np.where(one_hot_array > 160, 1, 0)

    # stitching images according to 1 hot upper triangle matrix
    x = ()
    initial = True
    # r, c = overlap_arr.shape
    for r in range(len(oa)):
        for c in range(len(oa[0])):
            if r < c and r != c and oa[r][c] == 1:
                if initial:

                    count2, x1 = (stitching(imgs[r], imgs[c]))
                    x = x1
                    initial = False
                else:

                    count3, x2 = stitching(x, imgs[c])
                    x = x2
    cv2.imwrite(savepath, x)
    x = ()
    initial = True
    # cv2.imshow('NEWIMAGE',x)
    # cv2.waitKey(0)
    return oa


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3',N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)










    #References:
    # https: // docs.opencv.org / 3.4 / dc / dc3 / tutorial_py_matcher.html

