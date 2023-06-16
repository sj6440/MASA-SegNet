import os

from PIL import Image
from tqdm import tqdm

from MASASeg import MASASeg
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    miou_mode       = 0
    num_classes     = 5
    #name_classes    = ["_background_","other","water","land use","natural","industrial","housing"]
    #name_classes    = ["_background_","housing","industrial","natural","land use","water","other"]
    name_classes    = ["vegetation","road","building","others","water"]
    #name_classes    = ["Montain","Water","Vegetation","High-Density Urban","Low-Density Urban"," Developd","_background_"]#PolSF-GF-3
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = ''

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = ""
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 :
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = MASASeg()
        #插值1
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision,FWIOUs = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes,FWIOUs)