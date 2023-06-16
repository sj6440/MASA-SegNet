## MASA-SegNet: A Semantic Segmentation Network for PolSAR Images
We provide PyTorch training code and prediction code for **MASA-SegNet**.
This code is **universal** for datasets of different polarization types.  
Meanwhile, image preprocessing methods for single-polarization and multi-polarization data has been released.  
##### 1. datasets  
In our study, we used two datasets to examine the effectiveness of MASA-SegNet. 
One is the single-polarization dataset [the FUSAR-Map](https://github.com/fudanxu/FUSAR-Map/). 
And the other is the quad-polarization dataset [the AIR-PolSAR-Seg](https://github.com/AICyberTeam/AIR-PolSAR-Seg).

#### 2. Training 
Please  put the original image data into the "Data" folder, and all our dataset files are formatted to support the VOC standard.  
Before you start training, please make sure that your dataset files are formatted correctly.  
There are two steps to train this model:
* Run the data preprocessing script on the training dataset.  
`python imgpre/Muti_SAR.py`  
`python imgpre/OP_SAR.py`  
* Run the train script on the training dataset.  
`python train.py --num_classes=7 --UnFreeze_Epoch=500 --batch_size=8`
#### 3. Test 
The get_miou.py file will help you calculate the performance of existing weights on the dataset using widely recognized metrics such as mAP, miou, and fwiou.
