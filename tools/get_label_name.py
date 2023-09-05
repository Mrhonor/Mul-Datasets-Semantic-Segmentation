labels_info = [
    {"name": "person", "id": 0, "color": [0, 0, 0], "trainId": 4},
    {"name": "truck", "id": 1, "color": [0, 0, 0], "trainId": 10},
    {"name": "fence", "id": 2, "color": [0, 0, 0], "trainId": 15},
    {"name": "billboard", "id": 3, "color": [0, 0, 0], "trainId": 17},
    {"name": "bus", "id": 4, "color": [0, 0, 0], "trainId": 11},
    {"name": "out of roi", "id": 5, "color": [0, 0, 0], "trainId": 5},
    {"name": "curb", "id": 6, "color": [0, 0, 0], "trainId": 13},
    {"name": "obs-str-bar-fallback", "id": 7, "color": [0, 0, 0], "trainId": 21},
    {"name": "tunnel", "id": 8, "color": [0, 0, 0], "trainId": 23},
    {"name": "non-drivable fallback", "id": 9, "color": [0, 0, 0], "trainId": 3},
    {"name": "bridge", "id": 10, "color": [0, 0, 0], "trainId": 23},
    {"name": "road", "id": 11, "color": [0, 0, 0], "trainId": 0},
    {"name": "wall", "id": 12, "color": [0, 0, 0], "trainId": 14},
    {"name": "traffic sign", "id": 13, "color": [0, 0, 0], "trainId": 18},
    {"name": "trailer", "id": 14, "color": [0, 0, 0], "trainId": 12},
    {"name": "animal", "id": 15, "color": [0, 0, 0], "trainId": 4},
    {"name": "building", "id": 16, "color": [0, 0, 0], "trainId": 22},
    {"name": "sky", "id": 17, "color": [0, 0, 0], "trainId": 25},
    {"name": "drivable fallback", "id": 18, "color": [0, 0, 0], "trainId": 1},
    {"name": "guard rail", "id": 19, "color": [0, 0, 0], "trainId": 16},
    {"name": "bicycle", "id": 20, "color": [0, 0, 0], "trainId": 7},
    {"name": "traffic light", "id": 21, "color": [0, 0, 0], "trainId": 19},
    {"name": "polegroup", "id": 22, "color": [0, 0, 0], "trainId": 20},
    {"name": "motorcycle", "id": 23, "color": [0, 0, 0], "trainId": 6},
    {"name": "car", "id": 24, "color": [0, 0, 0], "trainId": 9},
    {"name": "parking", "id": 25, "color": [0, 0, 0], "trainId": 1},
    {"name": "fallback background", "id": 26, "color": [0, 0, 0], "trainId": 25},
    {"name": "license plate", "id": 27, "color": [0, 0, 0], "trainId": 255},
    {"name": "rectification border", "id": 28, "color": [0, 0, 0], "trainId": 255},
    {"name": "train", "id": 29, "color": [0, 0, 0], "trainId": 255},
    {"name": "rider", "id": 30, "color": [0, 0, 0], "trainId": 5},
    {"name": "rail track", "id": 31, "color": [0, 0, 0], "trainId": 3},
    {"name": "sidewalk", "id": 32, "color": [0, 0, 0], "trainId": 2},
    {"name": "caravan", "id": 33, "color": [0, 0, 0], "trainId": 12},
    {"name": "pole", "id": 34, "color": [0, 0, 0], "trainId": 20},
    {"name": "vegetation", "id": 35, "color": [0, 0, 0], "trainId": 24},
    {"name": "autorickshaw", "id": 36, "color": [0, 0, 0], "trainId": 8},
    {"name": "vehicle fallback", "id": 37, "color": [0, 0, 0], "trainId": 12},
    {"name": "unlabel", "id":255, "color":[0,0,0], "trainId": 255},
]

names = []
for i in range(0, 26):
    name = None
    for lb in labels_info:
        if lb["trainId"] == i:
            if name is None:
                name = lb["name"]
            else:
                name = name + " or " + lb["name"]
    names.append(name)
            
    
print(names)