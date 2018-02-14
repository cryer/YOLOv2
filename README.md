# YOLOv2
Keras implementation of YOLOv2 refer to Andrew Ng

## Inspiration

This code mainly refers to deeplearning.ai by Andrew Ng.

## Pretrain Model

You also need a pretrained YOLOv2 model,you can get it from official YOLO website ,but here I have already uploaded a h5 file on
[Google Driver](https://drive.google.com/open?id=1v-V94VX2JWIrsDN4u8t9QUCMfzS6x08c).Feel free to download,and put it into model_data 
directory.

## Results

Seems some outputs are not always good.You maybe can train it by yourself to  get better,you need to download COCO datasets or others.
However, some people may not know that COCO is the most difficult dataset in detection filed.You can get 80% mAP in PASCAL VOC,but
may only get 30% mAP in COCO.This is because of COCO's big classes and large number of  bounding boxes in each image, including some very 
small objects.

![](https://github.com/cryer/YOLOv2/raw/master/out/test.jpg)

![](https://github.com/cryer/YOLOv2/raw/master/out/test2.jpg)

![](https://github.com/cryer/YOLOv2/raw/master/out/test3.jpg)

![](https://github.com/cryer/YOLOv2/raw/master/out/test5.jpg)

![](https://github.com/cryer/YOLOv2/raw/master/out/test8.jpg)
