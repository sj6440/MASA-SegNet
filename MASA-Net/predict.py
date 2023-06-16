from PIL import Image
import os
from MASASeg import MASASeg

if __name__ == "__main__":

    deeplab = MASASeg()
    mode = "predict"

    count           = True
    name_classes    = ["_background_","industrial","natural","land use","water","other","housing"]

    if mode == "predict":

        while True:
            imgname = os.listdir(r"\sar_predict")
            labname = os.listdir(r"\lab_predict")
            #img = input('Input image filename:')
            #label = input('input image label file:')
            for img in imgname:
                try:
                    image = Image.open(os.path.join(r"\sar_predict",img))
                    label = Image.open(os.path.join(r"\lab_predict",img[0:-4]+".png"))
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = deeplab.detect_image(image, label ,count=count, name_classes=name_classes)
                    r_image.save(os.path.join(r"E:\pythonProject\spacemodel\result_air_fusion",img))
