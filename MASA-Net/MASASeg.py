import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.decoder import Decoder
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# -----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
# -----------------------------------------------------------------------------------#
class MASASeg(object):
    _defaults = {

        "model_path": r'',

        "num_classes": 7,

        "backbone": "gMAXIMV2_3_backbone",

        "input_shape": [256, 256],

        "downsample_factor": 16,

        "mix_type": 1,

        "cuda": True,
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),(255, 255, 255)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)


    def generate(self, onnx=False):

        self.net = Decoder(num_classes=self.num_classes, backbone=self.backbone,
                           downsample_factor=self.downsample_factor, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()


    def fast_hist(self, a, b, n):

        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self, hist):
        return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

    def Frequency_Weighted_Intersection_over_Union(self, hist):
        freq = np.sum(hist, axis=1) / np.sum(hist)
        iu = np.diag(hist) / (
                np.sum(hist, axis=1) + np.sum(hist, axis=0) -
                np.diag(hist))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def detect_image(self, image, lab, count=False, name_classes=None):
        image = cvtColor(image)
        lab = cvtColor(lab)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            pr = pr.argmax(axis=-1)
            num_classes = len(name_classes)
            lab = np.array(lab)
            lab = lab[:, :, 0]
            hist = np.zeros((num_classes, num_classes))

            if len(lab.flatten()) != len(pr.flatten()):
                print(
                    'Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(
                        len(lab.flatten()), len(pr.flatten())))

            hist += self.fast_hist(lab.flatten(), pr.flatten(), num_classes)
            IoUs = self.per_class_iu(hist)
            FWIoU = self.Frequency_Weighted_Intersection_over_Union(hist)
            print("IOU:" + str(round(np.nanmean(IoUs) * 100, 2)) + '\n')
            print("FWIOU:" + str(round(np.nanmean(FWIoU) * 100, 2)) + '\n')

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image = Image.fromarray(np.uint8(seg_img))


        elif self.mix_type == 1:

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image = Image.fromarray(np.uint8(seg_img))


        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_miou_png(self, image):

        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()


            pr = self.net(images)[0]


            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)
            # pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_CUBIC)

            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
