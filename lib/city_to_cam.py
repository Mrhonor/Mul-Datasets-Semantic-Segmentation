import torch

labels_info = [
    {"name": "road", "trainId": 0, "Cam_name": "Road", "CamId": 3},
    {"name": "sidewalk", "trainId": 1, "Cam_name": "Sidewalk", "CamId": 4},
    {"name": "building", "trainId": 2, "Cam_name": "Building", "CamId": 1},
    {"name": "wall", "trainId": 3, "Cam_name": "Building", "CamId": 1},
    {"name": "fence", "trainId": 4, "Cam_name": "Fence", "CamId": 7},
    {"name": "pole", "trainId": 5, "Cam_name": "Column_Pole", "CamId": 2},
    {"name": "traffic light", "trainId": 6, "Cam_name": "SignSymbol", "CamId": 6},
    {"name": "traffic sign", "trainId": 7, "Cam_name": "SignSymbol", "CamId": 6},
    {"name": "vegetation", "trainId": 8, "Cam_name": "Tree", "CamId": 5},
    {"name": "terrain", "trainId": 9, "Cam_name": "Tree", "CamId": 5},
    {"name": "sky", "trainId": 10, "Cam_name": "Sky", "CamId": 0},
    {"name": "person", "trainId": 11, "Cam_name": "Pedestrian", "CamId": 9},
    {"name": "rider", "trainId": 12, "Cam_name": "Bicyclist", "CamId": 10},
    {"name": "car", "trainId": 13, "Cam_name": "Car", "CamId": 8},
    {"name": "truck", "trainId": 14, "Cam_name": "Car", "CamId": 8},
    {"name": "bus", "trainId": 15, "Cam_name": "Car", "CamId": 8},
    {"name": "train", "trainId": 16, "Cam_name": "Car", "CamId": 8},
    {"name": "motorcycle", "trainId": 17, "Cam_name": "Bicyclist", "CamId": 10},
    {"name": "bicycle", "trainId": 18, "Cam_name": "Bicyclist", "CamId": 10},
]

def Cityid_to_Camid(labels):
    mask = labels.clone()
    for el in labels_info:
        mask[labels == el["trainId"]] = el["CamId"]
        
    return mask

if __name__ == "__main__":
    a = torch.tensor([[0,11],[2,3]])
    print(a)
    print(Cityid_to_Camid(a))
    print(a)
