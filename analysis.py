# class_name = ["Sky", "Building", "Pole", "Road", "Sidewalk", "Tree", "traffic sign", "Fence", "Car", "Pedestrian", "Bicycl", "traffic light", "vegetation", "terrain"]

# out_lines = []
# for name in class_name:
#     this_out_lines = []
#     with open('label_mapping.log', 'r') as f:
#         for i,line in enumerate(f):
#             if name.lower() in line.lower():
#                 this_out_lines.append(i)
#     out_lines.append(this_out_lines)
        
# out_lines[5] = sorted(out_lines[5]+out_lines[-2]+out_lines[-1])
# out_lines[6] = sorted(out_lines[6]+out_lines[-3])
# with open('camvid_mapping2.txt', 'w') as fw:
#     for line in out_lines:
#         fw.write(str(line)+'\n')

with open('label_mapping.log', 'r') as file:
    lines = file.readlines()

# 遍历每行数据
out_lines = []
for line in lines:
    # 删除首尾的引号和空格，并分割成类别和得分部分
    line = line.strip('"\n').replace(' ', '').lower()
    parts = line.split('\',\'')
    
    category_scores = {}
    # 遍历每个类别和得分
    for part in parts:
        
        category, score_str = part.split(':')[-2:]
        
        
        if score_str == '[]':
            
            continue
        score = float(score_str.strip('\']'))
        
        # 检查字典中是否已有该类别，如果没有，则创建一个初始得分为0
        if category in category_scores:
            category_scores[category] += score
        else:
            category_scores[category] = score
            
    
    if len(category_scores) == 0:
        out_lines.append("None")
        continue
    
    maxkey = max(category_scores, key=category_scores.get)
    out_lines.append(maxkey)
    # for key, value in category_scores.items():
        
            
with open('label_names.txt', 'w') as fw:
    for line in out_lines:
        fw.write(str(line)+'\n')

# # 打印每个类别的总得分
# for category, total_score in category_scores.items():
#     print(f"{category}: {total_score}")