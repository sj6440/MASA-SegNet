import os
import cv2
import numpy as np
from scipy.signal import convolve2d

def refined_lee_filter(img, window_size=7, sigma=0.5):

    med = np.median(img)
    img = img - med

    # 计算滑动窗口的权重
    w = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            x = i - window_size // 2
            y = j - window_size // 2
            w[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # 计算每个像素的局部均值和方差
    mean = convolve2d(img, w, mode='same') / np.sum(w)
    var = convolve2d(img ** 2, w, mode='same') / np.sum(w) - mean ** 2

    # 根据局部均值和方差计算滤波后的像素值
    img_filt = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if var[i, j] > 0:
                img_filt[i, j] = mean[i, j] + max(0, var[i, j] - sigma ** 2) * (img[i, j] - mean[i, j]) / var[i, j]

    return img_filt + med

def frgb_img(img_hh,img_hv,img_vh,img_vv):
    img_hh = cv2.imread(img_hh, -1)
    img_hv = cv2.imread(img_hv, -1)
    img_vh = cv2.imread(img_vh, -1)
    img_vv = cv2.imread(img_vv, -1)
    img_hh = refined_lee_filter(img_hh, window_size=1, sigma=0.9)
    img_hv = refined_lee_filter(img_hv, window_size=1, sigma=0.9)
    img_vh = refined_lee_filter(img_vh, window_size=1, sigma=0.9)
    img_vv = refined_lee_filter(img_vv, window_size=1, sigma=0.9)

    img_rgb = np.zeros((512, 512, 3))

    b_img = 255 - img_hh
    r_img = 255 - img_vv
    g_img = 255 - ((img_hv + img_vh) / 2)

    img_rgb[:, :, 0] = r_img
    img_rgb[:, :, 1] = g_img
    img_rgb[:, :, 2] = b_img
    #img_rgb = cv2.normalize(img_rgb,None,0,255,cv2.NORM_INF)
    img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    return img_rgb
#
if __name__ == "__main__":
    source_root = r"Data"
    detis_root = r"Train_data/VOC2007/JPEGImages"

    classimage = []


    if not os.path.exists(source_root):
        os.mkdir(source_root)
    if not os.path.exists(detis_root):
        os.mkdir(detis_root)

    imagename = os.listdir(source_root)
    flag = 0
    for img in imagename:
        classimage.append(img)
        flag = flag + 1
        if flag == 5:
            print(classimage)
            imgname_HH = os.path.join(source_root, classimage[1])
            imgname_VV = os.path.join(source_root, classimage[4])
            imgname_HV = os.path.join(source_root, classimage[2])
            imgname_VH = os.path.join(source_root, classimage[3])


            image = frgb_img(imgname_HH,imgname_HV,imgname_VH,imgname_VV)
            image = np.array(image, np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            imgpath = os.path.join(detis_root, str(classimage[1][4:-8])) + ".jpg"
            cv2.imwrite(imgpath, image)
            img_color = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
            img_color = cv2.applyColorMap(img_color,cv2.COLORMAP_JET)
            cv2.imwrite(imgpath, img_color)

            flag = 0
            classimage = []
        else:
            continue




