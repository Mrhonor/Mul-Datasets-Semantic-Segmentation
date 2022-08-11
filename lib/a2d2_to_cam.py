import torch

# labels_info = [
#     {"name": "road", "trainId": 0, "Cam_name": "Road", "CamId": 3},
#     {"name": "sidewalk", "trainId": 1, "Cam_name": "Sidewalk", "CamId": 4},
#     {"name": "building", "trainId": 2, "Cam_name": "Building", "CamId": 1},
#     {"name": "wall", "trainId": 3, "Cam_name": "Building", "CamId": 1},
#     {"name": "fence", "trainId": 4, "Cam_name": "Fence", "CamId": 7},
#     {"name": "pole", "trainId": 5, "Cam_name": "Column_Pole", "CamId": 2},
#     {"name": "traffic light", "trainId": 6, "Cam_name": "SignSymbol", "CamId": 6},
#     {"name": "traffic sign", "trainId": 7, "Cam_name": "SignSymbol", "CamId": 6},
#     {"name": "vegetation", "trainId": 8, "Cam_name": "Tree", "CamId": 5},
#     {"name": "terrain", "trainId": 9, "Cam_name": "Tree", "CamId": 5},
#     {"name": "sky", "trainId": 10, "Cam_name": "Sky", "CamId": 0},
#     {"name": "person", "trainId": 11, "Cam_name": "Pedestrian", "CamId": 9},
#     {"name": "rider", "trainId": 12, "Cam_name": "Bicyclist", "CamId": 10},
#     {"name": "car", "trainId": 13, "Cam_name": "Car", "CamId": 8},
#     {"name": "truck", "trainId": 14, "Cam_name": "Car", "CamId": 8},
#     {"name": "bus", "trainId": 15, "Cam_name": "Car", "CamId": 8},
#     {"name": "train", "trainId": 16, "Cam_name": "Car", "CamId": 8},
#     {"name": "motorcycle", "trainId": 17, "Cam_name": "Bicyclist", "CamId": 10},
#     {"name": "bicycle", "trainId": 18, "Cam_name": "Bicyclist", "CamId": 10},
# ]

labels_info = [
    {"name": "Car", "trainId": 0, "Cam_name": "Car", "CamId": 8},
    {"name": "Bicycle", "trainId": 1, "Cam_name": "Bicyclist", "CamId": 10},
    {"name": "Pedestrian", "trainId": 2, "Cam_name": "Pedestrian", "CamId": 9},
    {"name": "Truck", "trainId": 3, "Cam_name": "Car", "CamId": 8},
    {"name": "Small vehicles", "trainId": 4, "Cam_name": "Car", "CamId": 8},
    {"name": "Traffic signal", "trainId": 5, "Cam_name": "SignSymbol", "CamId": 6},
    {"name": "Traffic sign", "trainId": 6, "Cam_name": "SignSymbol", "CamId": 6},
    {"name": "Utility vehicle", "trainId": 7, "Cam_name": "Car", "CamId": 8},
    {"name": "Sidebars", "trainId": 8, "Cam_name": "Fence", "CamId": 7},
    {"name": "Speed bumper", "trainId": 9, "Cam_name": "Road", "CamId": 3},
    {"name": "Curbstone", "trainId": 10, "Cam_name": "Sidewalk", "CamId": 4}, # 路中间和路边都有，无法很好覆盖
    {"name": "Solid line", "trainId": 11, "Cam_name": "Road", "CamId": 3},
    {"name": "Irrelevant signs", "trainId": 12, "Cam_name": "SignSymbol", "CamId": 6}, # 如背面的交通标志
    {"name": "Road blocks", "trainId": 13, "Cam_name": "Fence", "CamId": 7},
    {"name": "Tractor", "trainId": 14, "Cam_name": "Car", "CamId": 8},
    {"name": "Non-drivable street", "trainId": 15, "Cam_name": "Sidewalk", "CamId": 4},
    {"name": "Zebra crossing", "trainId": 16, "Cam_name": "Road", "CamId": 3},
    {"name": "Obstacles / trash", "trainId": 17, "Cam_name": "Building", "CamId": 1},
    {"name": "Poles", "trainId": 18, "Cam_name": "Column_Pole", "CamId": 2},
    {"name": "RD restricted area", "trainId": 19, "Cam_name": "Road", "CamId": 3},
    {"name": "Animals", "trainId": 20, "Cam_name": "Pedestrian", "CamId": 9},
    {"name": "Grid structure", "trainId": 21, "Cam_name": "Fence", "CamId": 7},
    {"name": "Signal corpus", "trainId": 22, "Cam_name": "SignSymbol", "CamId": 6},
    {"name": "Drivable cobblestone", "trainId": 23, "Cam_name": "Road", "CamId": 3},
    {"name": "Electronic traffic", "trainId": 24, "Cam_name": "SignSymbol", "CamId": 6},
    {"name": "Slow drive area", "trainId": 25, "Cam_name": "Road", "CamId": 3},
    {"name": "Nature object", "trainId": 26, "Cam_name": "Tree", "CamId": 5},
    {"name": "Parking area", "trainId": 27, "Cam_name": "Road", "CamId": 3},
    {"name": "Sidewalk", "trainId": 28, "Cam_name": "Sidewalk", "CamId": 4},
    {"name": "Ego car", "trainId": 29, "Cam_name": "Car", "CamId": 8},
    {"name": "Painted driv. instr.", "trainId": 30, "Cam_name": "Road", "CamId": 3},
    {"name": "Traffic guide obj.", "trainId": 31, "Cam_name": "Column_Pole", "CamId": 2},
    {"name": "Dashed line", "trainId": 32, "Cam_name": "Road", "CamId": 3},
    {"name": "RD normal street", "trainId": 33, "Cam_name": "Road", "CamId": 3},
    {"name": "Sky", "trainId": 34, "Cam_name": "Sky", "CamId": 0},
    {"name": "Buildings", "trainId": 35, "Cam_name": "Building", "CamId": 1},
    {"name": "Blurred area", "trainId": 36, "Cam_name": "Building", "CamId": 1},
    {"name": "Rain dirt", "trainId": 37, "Cam_name": "Road", "CamId": 3},
]

def a2d2_to_Camid(labels):
    mask = labels.clone()
    for el in labels_info:
        mask[labels == el["trainId"]] = el["CamId"]
        
    return mask

if __name__ == "__main__":
    a = torch.tensor([[0,11],[2,3]])
    print(a)
    print(Cityid_to_Camid(a))
    print(a)
