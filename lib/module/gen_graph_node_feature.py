import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from tools.configer import Configer
from lib.get_dataloader import get_data_loader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import clip
from PIL import Image
import lib.transform_cv2 as T
import torch.nn.functional as F

def get_img_for_everyclass(configer):
    n_datasets = configer.get("n_datasets")

    num_classes = []
    for i in range(1, n_datasets + 1):
        num_classes.append(configer.get("dataset" + str(i), "n_cats"))

    dl_city, dl_cam, dl_a2d2 = get_data_loader(configer, aux_mode='ret_path', distributed=False)

    city_img_lists = []
    city_lb_lists = []
    for label_id in range(0, num_classes[0]):
        city_img_lists.append([])
        city_lb_lists.append([])
        
    for im, lb in dl_city:
        for label_id in range(0, num_classes[0]):
            im = im[0]
            lb = lb.squeeze()
            if len(city_img_lists[label_id]) > 50 and (lb == label_id).any():
                city_img_lists[label_id].append(im)
                city_lb_lists[label_id].append(lb)

    cam_img_lists = []
    cam_lb_lists = []
    for label_id in range(0, num_classes[0]):
        cam_img_lists.append([])
        cam_lb_lists.append([])
        
    for im, lb in dl_cam:
        for label_id in range(0, num_classes[0]):
            im = im[0]
            lb = lb.squeeze()
            if len(cam_img_lists[label_id]) > 50 and (lb == label_id).any():
                cam_img_lists[label_id].append(im)
                cam_lb_lists[label_id].append(lb)

    a2d2_img_lists = []
    a2d2_lb_lists = []
    for label_id in range(0, num_classes[0]):
        a2d2_img_lists.append([])
        a2d2_lb_lists.append([])
        
    for im, lb in dl_a2d2:
        for label_id in range(0, num_classes[0]):
            im = im[0]
            lb = lb.squeeze()
            if len(a2d2_img_lists[label_id]) > 50 and (lb == label_id).any():
                a2d2_img_lists[label_id].append(im)
                a2d2_lb_lists[label_id].append(lb)

    
    return city_img_lists, city_lb_lists, cam_img_lists, cam_lb_lists, a2d2_img_lists, a2d2_lb_lists

def crop_image_by_label_value(img, label, label_value):
    # 将标签二值化
    binary = np.zeros_like(label)
    binary[label == label_value] = 255

    binary = cv2.convertScaleAbs(binary)
    
    # 执行闭运算操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 计算轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
#     # 找到覆盖所有轮廓的最小矩形
#     max_rect = cv2.minAreaRect(np.concatenate(contours))
    
#     # 获取最大包围盒的坐标
#     x, y, w, h = cv2.boundingRect(np.int0(cv2.boxPoints(max_rect)))
#     print(x, y, w, h)

    # 计算每个包围盒的面积并找到面积最大的包围盒
    max_area = 0
    max_bbox = None
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        area = bbox[2] * bbox[3]
        if area > max_area:
            max_area = area
            max_bbox = bbox

    # 如果没有找到任何包围盒，返回空图像
    if max_bbox is None:
        return np.zeros_like(img)

    # 裁剪图像
    x, y, w, h = max_bbox

#     # 获取包围盒
#     x, y, w, h = cv2.boundingRect(contours[4])
#     print(x, y, w, h)
    
    # 裁剪图像
    # print(img.shape)
    cropped = img[y:y+h, x:x+w, :]
    
    # 将不属于该标签的像素点替换为指定的值
    label_roi = binary[y:y+h, x:x+w]
    
    k = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(label_roi, kernel=k)
    
    
    mask = np.ones_like(cropped, dtype=bool)
    mask[dilated != 255] = False
    cropped[~mask] = 128
    
    h, w, _ = cropped.shape
    if h < w:
        top_padding = (w - h) // 2
        bottom_padding = w - h - top_padding
        cropped = cv2.copyMakeBorder(cropped, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    elif h > w:
        left_padding = (h - w) // 2
        right_padding = h - w - left_padding
        cropped = cv2.copyMakeBorder(cropped, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        
    # 返回裁剪后的图像
    return cropped


def gen_image_features(configer, city_img_lists, city_lb_lists, cam_img_lists, cam_lb_lists, a2d2_img_lists, a2d2_lb_lists):
    to_tensor = T.ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        city_image_features_lists = []
        for i, im_lb_list in enumerate(zip(city_img_lists, city_lb_lists)):
            im_list, lb_list = im_lb_list
            image_features_list = []
            for im_path, lb in zip(im_list, lb_list):
                image = cv2.imread(im_path)
                lb = lb.numpy()
                cropped_img = crop_image_by_label_value(image, lb, i)
                    
                im_lb = dict(im=cropped_img, lb=lb)
                im_lb = to_tensor(im_lb)
                img = im_lb['im'].cuda()
                img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                image_features = model.encode_image(img)
                image_features_list.append(image_features)
    # #             image_features_list.append(img)
    #         feats = torch.cat(image_features_list, dim=0)
    #         city_image_features_lists.append(feats.unsqueeze(0))
    #     feats = torch.cat(city_image_features_lists, dim=0)
    
    return

def get_encode_lb_vec(configer):
    n_datasets = configer.get('n_datasets')
    text_feature_vecs = []
    with torch.no_grad():
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
        for i in range(0, n_datasets):
            lb_name = configer.get("dataset"+str(i+1), "label_names")
            lb_name = ["a photo of " + name + "." for name in lb_name]
            text = clip.tokenize(lb_name).cuda()
            text_features = clip_model.encode_text(text).type(torch.float32)
            text_feature_vecs.append(text_features)
            
    return text_feature_vecs
                
def gen_graph_node_feature(configer):
    return
