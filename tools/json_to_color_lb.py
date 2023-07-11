import json

from PIL import Image, ImageDraw
lb_file = '../datasets/IDD/val.txt'

labels = set()
labels_info = [
{"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 0},
{"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 1},
{"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 2},
{"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 3},
{"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 4},
{"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 5},
{"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 6},
{"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 7},
{"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 8},
{"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 9},
{"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 10},
{"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 11},
{"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 12},
{"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 13},
{"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 14},
{"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 15},
{"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 16},
{"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 17},
{"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 18},
{"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 19},
{"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 20},
{"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 21},
{"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 22},
{"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 23},
{"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 24},
{"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 25},
{"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 26},
{"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 27},
{"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 28},
{"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 29},
{"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 30},
{"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 31},
{"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 32},
{"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 33},
{"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 34},
{"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 35},
{"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 36},
{"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 37}
]

labels_info_map = {}
for lb_if in labels_info:
    labels_info_map[lb_if['name']] = lb_if['id']

with open(lb_file, 'r') as f:
    for line in f:
        # im_path = line.split(',')[0]
        lb_path = '/home1/marong/datasets/idd/' + line.split(',')[1].replace('png', 'json').replace('\n', '')
        with open(lb_path, 'r') as json_f:
            data = json.load(json_f)
            imH = data['imgHeight']
            imW = data['imgWidth']
            annotation_image = Image.new('L', (imW, imH), 255)
            draw = ImageDraw.Draw(annotation_image)
            for annotation in data['objects']:
                if annotation['deleted'] != 0 or annotation['draw'] != True or annotation['label'] == 'ego vehicle':
                    continue

                label = annotation['label']
                polygons = annotation['polygon']
                polygons = [(int(point[0]), int(point[1])) for point in polygons]
                # print(polygons)
                # labels.add(label)
                draw.polygon(polygons, fill=labels_info_map[label])
            annotation_image.save(lb_path.replace('.json', '.png'))






                # print (annotation)
# print ("!")
# print (labels)
# for lb in labels:
#     print (lb)
            #

#

#
# # 读取原始图像

#
# # 创建空白的逐像素点标注图像
# annotation_image = Image.new('L', (width, height), 0)
# draw = ImageDraw.Draw(annotation_image)
#
# # 绘制多边形
# draw.polygon(vertices, fill=255)
#
# # 保存逐像素点标注图像
# annotation_image.save('annotation.png')
