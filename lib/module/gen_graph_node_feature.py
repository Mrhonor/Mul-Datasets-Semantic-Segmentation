import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

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
import random
import pickle

def get_img_for_everyclass(configer, dataset_id=None):
    n_datasets = configer.get("n_datasets")

    num_classes = []
    for i in range(1, n_datasets + 1):
        num_classes.append(configer.get("dataset" + str(i), "n_cats"))

    dls = get_data_loader(configer, aux_mode='ret_path', distributed=False)

    img_lists = []
    lb_lists  = []
    lb_info_list = []
    for i in range(0, n_datasets):
        this_img_lists = []
        this_lb_lists = []
        lb_info_list.append(dls[i].dataset.lb_map)
        
        if dataset_id != None and i != dataset_id:
            img_lists.append(this_img_lists)
            lb_lists.append(this_lb_lists)
            continue
        # print("cur dataset id： ", i)
        
        for label_id in range(0, num_classes[i]):
            this_img_lists.append([])
            this_lb_lists.append([])
            
        for im, lb, lbpth in dls[i]:
            im = im[0]
            lb = lb.squeeze()
            for label_id in range(0, num_classes[i]):
                if len(this_img_lists[label_id]) < 100 and (lb == label_id).any():
                    this_img_lists[label_id].append(im)
                    this_lb_lists[label_id].append(lbpth)
                    

        for j, lb in enumerate(this_lb_lists):
            if len(lb) == 0:
                print("the number {} class has no image".format(j))
        
        
        img_lists.append(this_img_lists)
        lb_lists.append(this_lb_lists)
        
    
    return img_lists, lb_lists, lb_info_list

def get_img_for_everyclass_single(configer, dls):
    n_datasets = configer.get("n_datasets")

    num_classes = []
    for i in range(1, n_datasets + 1):
        num_classes.append(configer.get("dataset" + str(i), "n_cats"))

    # dls = get_data_loader(configer, aux_mode='ret_path', distributed=False)
    dl_iters = [iter(dl) for dl in dls]

    img_lists = []
    lb_lists  = []
    for i in range(0, n_datasets):
        # print("cur dataset id： ", i)
        this_img_lists = []
        this_lb_lists = []
        for label_id in range(0, num_classes[i]):
            this_img_lists.append([])
            this_lb_lists.append([])
            
        cur_num = 0
        while True:
            try:
                im, lb, lbpth = next(dl_iters[i])
                while torch.min(lb) == 255:
                    im, lb, lbpth = next(dl_iters[i])

                if not len(im) == 1:
                    raise StopIteration
            except StopIteration:
                break
            

            im = im[0]
            lb = lb.squeeze()
            for label_id in range(0, num_classes[i]):
                if (lb == label_id).any():
                    this_img_lists[label_id].append(im)
                    this_lb_lists[label_id].append(lbpth)
                    
            # print('cur_num: ', cur_num)
            # print('num_classes: ', num_classes[i])
            # print(lbpth)
            # if cur_num == num_classes[i]:
            #     break

        for j, lb in enumerate(this_lb_lists):
            if len(lb) == 0:
                print("the number {} class has no image".format(j))
        
            
        img_lists.append(this_img_lists)
        lb_lists.append(this_lb_lists)

    return img_lists, lb_lists

def crop_image_by_label_value(img, label, label_value):
    # 将标签二值化
    binary = np.zeros_like(label, dtype=np.uint8)
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


def gen_image_features(configer, dataset_id=None):
    img_lists, lb_lists, lb_info_list = get_img_for_everyclass(configer, dataset_id)
    
    n_datasets = configer.get('n_datasets')
    to_tensor = T.ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        out_features = []
        for idx in range(0, n_datasets):
            this_label_info = lb_info_list[idx]
            if dataset_id != None and idx != dataset_id:
                continue
            print("dataset_id: ", idx)
            for i, im_lb_list in enumerate(zip(img_lists[idx], lb_lists[idx])):
                im_list, lb_list = im_lb_list
                if len(im_list) == 0:
                    print("why dataset_id: ", idx)
                    continue
                image_features_list = []
                for im_path, lb_path in zip(im_list, lb_list):
                    image = cv2.imread(im_path)
                    # print(lb_path[0])
                    lb = cv2.imread(lb_path[0], 0)
                    lb = this_label_info[lb]
                    if image is None:
                        print(im_path)
                        continue
                    # lb = lb.numpy()
                    cropped_img = crop_image_by_label_value(image, lb, i)
                        
                    im_lb = dict(im=cropped_img, lb=lb)
                    im_lb = to_tensor(im_lb)
                    img = im_lb['im'].cuda()
                    img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                    image_features = model.encode_image(img).type(torch.float32)
                    image_features_list.append(image_features)

                # print("im_lb_list: ", im_lb_list)
                img_feat = torch.cat(image_features_list, dim=0)
                mean_feats = torch.mean(img_feat, dim=0, keepdim=True)
                # print(mean_feats.shape)
                out_features.append(mean_feats) 
    
    return out_features

def gen_image_features_single(configer, dls, gen_feature=False):
    if gen_feature is False:
        img_lists, lb_lists = get_img_for_everyclass_single(configer, dls)
    else:
        img_lists, lb_lists = dls
        
    n_datasets = configer.get('n_datasets')
    to_tensor = T.ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        out_features = []
        for dataset_id in range(0, n_datasets):
            print("dataset_id: ", dataset_id)
            for i, im_lb_list in enumerate(zip(img_lists[dataset_id], lb_lists[dataset_id])):
                im_list, lb_list = im_lb_list
                im_lb_zip_list = list(zip(im_list, lb_list))
                if len(im_list) == 0:
                    print("why dataset_id: ", dataset_id)
                    continue
                image_features_list = []
                im_path, lb_path = random.choice(im_lb_zip_list)
                image = cv2.imread(im_path)
                # print(lb_path[0])
                lb = cv2.imread(lb_path[0], 0)
                if image is None:
                    print(im_path)
                    continue
                # lb = lb.numpy()
                cropped_img = crop_image_by_label_value(image, lb, i)
                    
                im_lb = dict(im=cropped_img, lb=lb)
                im_lb = to_tensor(im_lb)
                img = im_lb['im'].cuda()
                img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                image_features = model.encode_image(img).type(torch.float32)
                image_features_list.append(image_features)

                # print("im_lb_list: ", im_lb_list)
                img_feat = torch.cat(image_features_list, dim=0)
                mean_feats = torch.mean(img_feat, dim=0, keepdim=True)
                # print(mean_feats.shape)
                out_features.append(mean_feats) 

    
    return out_features, [img_lists, lb_lists]

def gen_image_features_storage(configer, dataset_id):
    img_lists, lb_lists = get_img_for_everyclass(configer, dataset_id)
    
    n_datasets = configer.get('n_datasets')
    to_tensor = T.ToTensor(
                mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
                std=(0.26862954, 0.26130258, 0.27577711),
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        this_datasets_feats = []
        print("dataset_id: ", dataset_id)
        for i, im_lb_list in enumerate(zip(img_lists[dataset_id], lb_lists[dataset_id])):
            im_list, lb_list = im_lb_list
            if len(im_list) == 0:
                print("why dataset_id: ", dataset_id)
                continue
            image_features_list = []
            for im_path, lb_path in zip(im_list, lb_list):
                image = cv2.imread(im_path)
                # print(lb_path[0])
                lb = cv2.imread(lb_path[0], 0)
                if image is None:
                    print(im_path)
                    continue
                # lb = lb.numpy()
                cropped_img = crop_image_by_label_value(image, lb, i)
                    
                im_lb = dict(im=cropped_img, lb=lb)
                im_lb = to_tensor(im_lb)
                img = im_lb['im'].cuda()
                img = F.interpolate(img.unsqueeze(0), size=(224, 224))
                image_features = model.encode_image(img).type(torch.float32)
                image_features_list.append(image_features)

            # print("im_lb_list: ", im_lb_list)
            img_feat = torch.cat(image_features_list, dim=0)
            # mean_feats = torch.mean(img_feat, dim=0, keepdim=True)
            # print(mean_feats.shape)
            this_datasets_feats.append(img_feat) 
    # out_features = torch.cat(out_features, dim=0)    
    return this_datasets_feats


def get_encode_lb_vec(configer, datasets_id=None):
    n_datasets = configer.get('n_datasets')
    text_feature_vecs = []
    with torch.no_grad():
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
        for i in range(0, n_datasets):
            if datasets_id != None and i != datasets_id:
                continue
            lb_name = configer.get("dataset"+str(i+1), "label_names")
            lb_name = ["a photo of " + name + "." for name in lb_name]
            text = clip.tokenize(lb_name).cuda()
            text_features = clip_model.encode_text(text).type(torch.float32)
            text_feature_vecs.append(text_features)
            
    return text_feature_vecs
                
def gen_graph_node_feature(configer):
    n_datasets = configer.get("n_datasets")

    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))
    
    file_name = configer.get('res_save_pth') + 'graph_node_features'
    dataset_names = []
    for i in range(0, configer.get('n_datasets')):
        # file_name += '_'+str(configer.get('dataset'+str(i+1), 'data_reader'))
        dataset_names.append(str(configer.get('dataset'+str(i+1), 'data_reader')))
    
    # file_name += '.pt'
    out_features = []
    for i in range(0, n_datasets):
        this_file_name = file_name + f'_{dataset_names[i]}.pt' 
        if osp.exists(this_file_name):
            this_graph_node_features = torch.load(this_file_name, map_location='cpu')

            out_features.append(this_graph_node_features)
        else:
            print(f'gen_graph_node_featuer: {i}')
            img_feature_vecs = gen_image_features(configer, i)
            img_feat_tensor = torch.cat(img_feature_vecs, dim=0)
            
            text_feature_vecs = get_encode_lb_vec(configer, i)[0]
            this_graph_node_features = torch.cat([text_feature_vecs, img_feat_tensor], dim=1)
            
            print("gen finished")
            torch.save(this_graph_node_features.clone(), this_file_name)

            out_features.append(this_graph_node_features)
    
    out_features = torch.cat(out_features, dim=0)
    print(out_features.shape)
    return out_features 
    
    
    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))
    
    file_name = configer.get('res_save_pth') + 'graph_node_features'+str(configer.get('n_datasets'))
    for i in range(0, configer.get('n_datasets')):
        file_name += '_'+str(configer.get('dataset'+str(i+1), 'data_reader'))
    
    file_name += '.pt'

    if osp.exists(file_name):
        graph_node_features = torch.load(file_name)
    else:
        print("gen_graph_node_feature")
        text_feature_vecs = get_encode_lb_vec(configer)
        text_feat_tensor = torch.cat(text_feature_vecs, dim=0)
        print(text_feat_tensor.shape)
        print("gen_text_feature_vecs")
        img_feature_vecs = gen_image_features(configer)
        img_feat_tensor = torch.cat(img_feature_vecs, dim=0)
        print(img_feat_tensor.shape)
        print("gen_img_features")
        graph_node_features = torch.cat([text_feat_tensor, img_feat_tensor], dim=1)
        # graph_node_features = (text_feat_tensor+img_feat_tensor)/2
        print(graph_node_features.shape)
        torch.save(graph_node_features.clone(), file_name)
    
    return graph_node_features

    
def gen_graph_node_feature_single(configer, img_feature_vecs, text_feat_tensor=None):
    if text_feat_tensor is None:
        text_feature_vecs = get_encode_lb_vec(configer)
        text_feat_tensor = torch.cat(text_feature_vecs, dim=0)

    # img_feature_vecs, lbpth_list = gen_image_features_single(configer, dls, gen_feature)
    n_datasets = configer.get("n_datasets")
    img_feat_tensor = []
    for i in range(0, n_datasets):
        n_cats = configer.get(f'dataset{i+1}', 'n_cats')
        for j in range(0, n_cats):
            num_samples, dim = img_feature_vecs[i][j].shape
            if num_samples == 1:
                img_feat_tensor.append(img_feature_vecs[i][j])
                continue
                
            choice_samples = int(num_samples/2)
            
            # print(f'the shape of dataset{i}: ', img_feature_vecs[i][j].shape)
            random_img_sample = random.sample(list(img_feature_vecs[i][j]), choice_samples)
            avg_img_feat = torch.mean(torch.cat(random_img_sample).view(-1, dim), dim=0)
            img_feat_tensor.append(avg_img_feat[None])
    
        
    img_feat_tensor = torch.cat(img_feat_tensor, dim=0)

    # graph_node_features = torch.cat([text_feat_tensor, img_feat_tensor], dim=1)
    graph_node_features = (text_feat_tensor+img_feat_tensor)/2

    return graph_node_features, text_feat_tensor

def gen_graph_node_feature_storage(configer):

    # text_feature_vecs = get_encode_lb_vec(configer)
    # text_feat_tensor = torch.cat(text_feature_vecs, dim=0)
    n_datasets = configer.get("n_datasets")

    if not osp.exists(configer.get('res_save_pth')): os.makedirs(configer.get('res_save_pth'))
    
    file_name = configer.get('res_save_pth') + 'img_feature_vecs'+str(configer.get('n_datasets'))
    dataset_names = []
    for i in range(0, configer.get('n_datasets')):
        # file_name += '_'+str(configer.get('dataset'+str(i+1), 'data_reader'))
        dataset_names.append(str(configer.get('dataset'+str(i+1), 'data_reader')))
    
    # file_name += '.pt'
    out_features = []
    for i in range(0, n_datasets):
        this_file_name = file_name + f'_{dataset_names[i]}.pkl' 
        if osp.exists(this_file_name):
            # img_feature_vecs = torch.load(this_file_name)
            with open(this_file_name, 'rb') as file:
                img_feature_vecs = pickle.load(file)  

            out_features.append(img_feature_vecs)
        else:
            print("gen_img_feature_vecs")
            img_feature_vecs = gen_image_features_storage(configer, i)
            print("gen finished")
            with open(this_file_name, 'wb') as file:
                pickle.dump(img_feature_vecs, file)
            # _ = [torch.save(img_feature_vecs.clone(), file_name + f'_dataset{idx}.pt' ) for idx, img_feature_vecs in enumerate(out_features)]
            out_features.append(img_feature_vecs)
    
    return out_features 


if __name__ == "__main__":
    configer = Configer(configs="configs/ltbgnn_5_datasets.json")
    img_feature_vecs = gen_graph_node_feature_storage(configer) 
    # print(img_feature_vecs[0][0])
    # print(img_feature_vecs)

    # print(graph_node_features.shape)
    # norm_adj_feat = F.normalize(graph_node_features, p=2, dim=1)
    # similar_matrix = torch.einsum('nc, mc -> nm', norm_adj_feat, norm_adj_feat)
    # print("similar_matrix_max:", torch.max(similar_matrix))
    # print("similar_matrix_min:", torch.min(similar_matrix))
    # torch.set_printoptions(profile="full")
    # print(similar_matrix)
    
    