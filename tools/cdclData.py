#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import pandas as pd
from sklearn.utils import shuffle


def getPath(root, imgs, labels):
    contents = os.listdir(root)
    for content in contents:
        contPath = os.path.join(root, content)
        if(os.path.isfile(contPath)):
            if contPath.count('label') > 0:
                labels.append(contPath)
                imgs.append(contPath.replace('label', 'camera'))
        else:
            getPath(contPath, imgs, labels)

def getPathCityscapes(root, imgs, labels):
    contents = os.listdir(root)
    for content in contents:
        contPath = os.path.join(root, content)
        if(os.path.isfile(contPath)):
            imgs.append(contPath)
            labels.append(contPath.replace('leftImg8bit/', 'gtFine_trainvaltest/gtFine/').replace('_leftImg8bit', '_gtFine_labelIds'))
        else:
            getPathCityscapes(contPath, imgs, labels)

def getPathCamVid(root, imgs, labels):
    contents = os.listdir(root)
    for content in contents:
        contPath = os.path.join(root, content)
        if(os.path.isfile(contPath)):
            imgs.append(contPath)
            labels.append(contPath.replace('val/', 'val_labels/').replace('train/', 'train_labels/').replace('test', 'test_labels').replace('.png', '_L.png'))
        else:
            getPathCamVid(contPath, imgs, labels)

if __name__ == "__main__":
    # if not os.path.exists('/root/autodl-tmp/project/BiSeNet/datasets/A2D2'):
    #     os.mkdir('/root/autodl-tmp/project/BiSeNet/datasets/A2D2')
    #     images, labels = [], []
    #     getPath(root='/root/autodl-tmp/datasets/a2d2/camera_lidar_semantic', imgs=images, labels=labels)
    #     all = pd.DataFrame({'image': images, 'label': labels})
    #     #all = all.sort_values(by='image', ascending=True)
    #     nine_part = int(len(images) * 0.9)
    #     allshuffled = shuffle(all)
    #     train_list = allshuffled[:nine_part]
    #     val_list = allshuffled[nine_part:]
    #     train_list.to_csv('/root/autodl-tmp/project/BiSeNet/datasets/A2D2/train.txt', index=False)
    #     val_list.to_csv('/root/autodl-tmp/project/BiSeNet/datasets/A2D2/val.txt', index=False)

    # if not os.path.exists('/root/autodl-tmp/project/BiSeNet/datasets/Cityscapes'):
    #     os.mkdir('/root/autodl-tmp/project/BiSeNet/datasets/Cityscapes')

    #     images, labels = [], []
    #     getPathCityscapes(root='/root/autodl-tmp/datasets/CityScapes/leftImg8bit/train', imgs=images, labels=labels)
    #     train_list = pd.DataFrame({'image': images, 'label': labels})
    #     #all = all.sort_values(by='image', ascending=True)
    #     train_list = shuffle(train_list)
    #     train_list.to_csv('/root/autodl-tmp/project/BiSeNet/datasets/Cityscapes/train.txt', index=False)

    #     images, labels = [], []
    #     getPathCityscapes(root='/root/autodl-tmp/datasets/CityScapes/leftImg8bit/val', imgs=images, labels=labels)
    #     val_list = pd.DataFrame({'image': images, 'label': labels})
    #     # all = all.sort_values(by='image', ascending=True)
    #     val_list = shuffle(val_list)
    #     val_list.to_csv('/root/autodl-tmp/project/BiSeNet/datasets/Cityscapes/val.txt', index=False)

    #     images, labels = [], []
    #     getPathCityscapes(root='/root/autodl-tmp/datasets/CityScapes/leftImg8bit/test', imgs=images, labels=labels)
    #     test_list = pd.DataFrame({'image': images, 'label': labels})
    #     # all = all.sort_values(by='image', ascending=True)
    #     test_list = shuffle(test_list)
    #     test_list.to_csv('/root/autodl-tmp/project/BiSeNet/datasets/Cityscapes/test.txt', index=False)
        
    if not os.path.exists('../datasets/CamVid'):
        os.mkdir('../datasets/CamVid')

    images1, labels1 = [], []
    getPathCamVid(root='D:/Study/code/archive/CamVid/train/', imgs=images1, labels=labels1)
    train_list = pd.DataFrame({'image': images1, 'label': labels1})
    #all = all.sort_values(by='image', ascending=True)
    train_list = shuffle(train_list)
    train_list.to_csv('../datasets/CamVid/train.txt', index=False)

    images2, labels2 = [], []
    getPathCamVid(root='D:/Study/code/archive/CamVid/val/', imgs=images2, labels=labels2)
    val_list = pd.DataFrame({'image': images2, 'label': labels2})
    # all = all.sort_values(by='image', ascending=True)
    val_list = shuffle(val_list)
    val_list.to_csv('../datasets/CamVid/val.txt', index=False)

    images3, labels3 = [], []
    getPathCamVid(root='D:/Study/code/archive/CamVid/test/', imgs=images3, labels=labels3)
    test_list = pd.DataFrame({'image': images3, 'label': labels3})
    # all = all.sort_values(by='image', ascending=True)
    test_list = shuffle(test_list)
    test_list.to_csv('../datasets/CamVid/test.txt', index=False)
    
    all_list = pd.DataFrame({'image': images1+images2+images3, 'label': labels1+labels2+labels3})
    all_list = shuffle(all_list)
    all_list.to_csv('../datasets/CamVid/all.txt', index=False)