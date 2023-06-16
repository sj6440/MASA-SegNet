import os

import cv2
import numpy as np
from osgeo import gdal


def normalization_rgb(r_file, g_file, b_file):
    img_shape = r_file.shape
    img_all = np.zeros((img_shape[0], img_shape[1],3))

    img_all[:,:,0] = r_file
    img_all[:,:,1] = g_file
    img_all[:,:,2] = b_file

    print(img_all.shape)
    img_all = np.around(img_all)

    img_all = img_all.astype('uint8')

    return img_all

if __name__ == "__main__":
    source_root = r""
    opt_root = r""
    detis_root  = r""
    if not  os.path.exists(detis_root):
        os.mkdir(detis_root)


    img_name = os.listdir(source_root)
    opt_name = os.listdir(opt_root)
    for i in range(len(img_name)) :
        img_path = os.path.join(source_root,img_name[i])
        opt_path = os.path.join(opt_root,opt_name[i])
        img = gdal.Open(img_path)
        opt = gdal.Open(opt_path)
        tmp_img = img.ReadAsArray()
        tmp_opt = opt.ReadAsArray()

        g = tmp_opt[0,::]
        r = tmp_img
        b = tmp_img
        image = normalization_rgb(r,g,b)

        imgpath =os.path.join(detis_root,str(img_name[i][0:-4]))+".jpg"

        print(imgpath)
        cv2.imwrite(imgpath, image)




