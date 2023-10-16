import torch
import torch.nn as nn
from tools.configer import Configer
from lib.get_dataloader import get_data_loader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import clip
from PIL import Image
import lib.transform_cv2 as T
import torch.nn.functional as F

configer = Configer(configs='configs/clip_city_cam_a2d2.json')
# clip_model, _ = clip.load("ViT-B/32", device="cuda")

# lb_name = configer.get("unify_classes_name")
# lb_name = ["a photo of " + name + "." for name in lb_name]
# text = clip.tokenize(lb_name).cuda()
# text_features = clip_model.encode_text(text).type(torch.float32)

n_datasets = configer.get("n_datasets")

num_classes = []
for i in range(1, n_datasets + 1):
    num_classes.append(configer.get("dataset" + str(i), "n_cats"))

dl_city, dl_cam, dl_a2d2 = get_data_loader(configer, aux_mode='ret_path', distributed=False)

city_img_lists = []
city_lb_lists = []
for label_id in range(0, num_classes[0]):
    img_count = 0

    city_img_list = []
    city_lb_list = []
    for im, lb in dl_city:
        im = im[0]
        lb = lb.squeeze()
        if (lb == label_id).any():
            city_img_list.append(im)
            city_lb_list.append(lb)
            img_count += 1
            if img_count == 10:
                break
    city_img_lists.append(city_img_list)
    city_lb_lists.append(city_lb_list)


# cam_img_lists = []
# cam_lb_lists = []
# for label_id in range(0, num_classes[1]):
#     img_count = 0
#     cam_img_list = []
#     cam_lb_list = []
#     for im, lb in dl_cam:
#         if (lb == label_id).any():
#             cam_img_list.append(im)
#             cam_lb_list.append(lb)
#             img_count += 1
#             if img_count == 10:
#                 break
#     cam_img_lists.append(cam_img_list)
#     cam_lb_lists.append(cam_lb_list)

# a2d2_img_lists = []
# a2d2_lb_lists = []
# for label_id in range(0, num_classes[2]):
#     img_count = 0
#     a2d2_img_list = []
#     a2d2_lb_list = []
#     for im, lb in dl_a2d2:
#         if (lb == label_id).any():
#             a2d2_img_list.append(im)
#             a2d2_lb_list.append(lb)
#             img_count += 1
#             if img_count == 10:
#                 break
#     a2d2_img_lists.append(a2d2_img_list)
#     a2d2_lb_list.append(a2d2_lb_list)
    

# cv2.imshow('Cropped Image', cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


    
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
        return np.zeros_like(image)

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

im_path = city_img_lists[0][0]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

image = cv2.imread(im_path)

lb = city_lb_lists[0][0].numpy()

to_tensor = T.ToTensor(
            mean=(0.48145466, 0.4578275, 0.40821073), # clip , rgb
            std=(0.26862954, 0.26130258, 0.27577711),
        )


cropped_img = crop_image_by_label_value(image, lb, 5)

im_lb = dict(im=cropped_img, lb=lb)
im_lb = to_tensor(im_lb)
img = im_lb['im'].cuda()
# _, h, w = img.shape
# print(img.shape)
img = F.interpolate(img.unsqueeze(0), size=(224, 224))
# if h > w:
#     resize_w = int(224 * w / h)
#     img = F.interpolate(img.unsqueeze(0), size=(224, resize_w))
#     left_padding = (224 - resize_w) /2
#     right_padding = 224 - resize_w - left_padding
#     pad = nn.ZeroPad2d(padding=(left_padding, right_padding, 0, 0), value=0)
# else:
#     img = F.interpolate(img.unsqueeze(0), size=(int(224 * h / w), 224))
    


# lb = im_lb['lb']

text = clip.tokenize(["road", "a dog", "a cat", "a pole"]).cuda()

with torch.no_grad():
    # image_features = model.encode_image(img)
    # text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(img, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  

# # 显示裁剪前后的图像
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(im)
# axes[0].set_title('Original Image')
# axes[1].imshow(cropped_img)
# axes[1].set_title('Cropped Image')
# plt.show()

def pairwise_cosine(data1, data2, device='cpu'):
    # transfer to device
    if device == 'cuda':
        data1, data2 = data1.cuda(), data2.cuda()

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized
    
    # return N*N matrix for pairwise distance
    cosine_dis = cosine.sum(dim=-1) #.squeeze()
    return cosine_dis

                                 

# sim_matrix = pairwise_cosine(text_features, text_features)

with torch.no_grad():
    image_features_lists = []
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
        feats = torch.cat(image_features_list, dim=0)
        image_features_lists.append(feats.unsqueeze(0))
    feats = torch.cat(image_features_lists, dim=0)
    print(feats.shape)
        
    for i in range(len(image_features_lists)):
        before_feats = feats[:i, 0, :].squeeze()
        after_feats = feats[i+1:, 0, :].squeeze()
        expand_feat = torch.cat([before_feats, feats[i], after_feats], dim=0)
        
        sim_matrix = pairwise_cosine(expand_feat, expand_feat)
        print(sim_matrix)
        print(sim_matrix.shape)
        break
    
            
        
        
