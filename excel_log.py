import pandas as pd
log_file = '7_datasets_lr5e-4_cuda6.log'
d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
d7 = []
city_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
mapi_lb = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
sun_lb = [ "bag", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub"]
bdd_lb = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
idd_lb = ["road", "drivable fallback or parking", "sidewalk", "non-drivable fallback or rail track", "person or animal", "out of roi or rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "trailer or caravan or vehicle fallback", "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light", "polegroup or pole", "obs-str-bar-fallback", "building", "tunnel or bridge", "vegetation", "sky or fallback background"]
ade_lb = ["flag", "wall", "building, edifice", "sky", "floor, flooring", "tree", "ceiling", "road, route", "bed ", "windowpane, window ", "grass", "cabinet", "sidewalk, pavement", "person, individual, someone, somebody, mortal, soul", "earth, ground", "door, double door", "table", "mountain, mount", "plant, flora, plant life", "curtain, drape, drapery, mantle, pall", "chair", "car, auto, automobile, machine, motorcar", "water", "painting, picture", "sofa, couch, lounge", "shelf", "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair", "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs, steps", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table, cocktail table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove, kitchen stove, range, kitchen range, cooking stove", "palm, palm tree", "kitchen island", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "towel", "light, light source", "truck, motortruck", "tower", "chandelier, pendant, pendent", "awning, sunshade, sunblind", "streetlight, street lamp", "booth, cubicle, stall, kiosk", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "airplane, aeroplane, plane", "dirt track", "apparel, wearing apparel, dress, clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "swimming pool, swimming bath, natatorium", "stool", "barrel, cask", "basket, handbasket", "waterfall, falls", "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name, brand name, brand, marque", "microwave, microwave oven", "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna", "bicycle, bike, wheel, cycle ", "lake", "dishwasher, dish washer, dishwashing machine", "screen, silver screen, projection screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light, traffic signal, stoplight", "tray", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "fan", "pier, wharf, wharfage, dock", "crt screen", "plate", "monitor, monitoring device", "bulletin board, notice board", "shower", "radiator", "glass, drinking glass", "clock"]
coco_lb = ["rug-merged", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged"]

with open(log_file, 'r') as file:
    id = 1
    inside_tensor = False
    for line in file:
        if 'tensor([' in line:
            # 进入tensor区域
            inside_tensor = True
            start_index = line.find('[') + 1
            tensor_str = line[start_index:]
        elif inside_tensor and ']' in line:
            # 离开tensor区域
            end_index = line.find(']')
            tensor_str += line[:end_index]

            # 将字符串转换为实际的列表
            eval(f'd{id}').append([float(x) for x in tensor_str.split(',')])
            
            # 重置标志
            inside_tensor = False
        elif inside_tensor:
            # 在tensor区域内的中间行，直接添加内容
            tensor_str += line
            id += 1
            if id == 8:
                id = 1

data_map = {}
for name, value in zip(city_lb, d1[-1]):
    data_map[name] = value
print(data_map)    
df = pd.DataFrame(data_map)

print(df)

data_map = {}
for name, value in zip(mapi_lb, d2[-1]):
    data_map[name] = value
    
df = pd.DataFrame(data_map)

print(df)

data_map = {}
for name, value in zip(sun_lb, d3[-1]):
    data_map[name] = value
    
df = pd.DataFrame(data_map)

print(df)

data_map = {}
for name, value in zip(bdd_lb, d4[-1]):
    data_map[name] = value
    
df = pd.DataFrame(data_map)

print(df)

data_map = {}
for name, value in zip(idd_lb, d5[-1]):
    data_map[name] = value
    
df = pd.DataFrame(data_map)

print(df)

data_map = {}
for name, value in zip(ade_lb, d6[-1]):
    data_map[name] = value
    
df = pd.DataFrame(data_map)

print(df)

data_map = {}
for name, value in zip(coco_lb, d7[-1]):
    data_map[name] = value
    
df = pd.DataFrame(data_map)

print(df)


