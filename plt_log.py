import matplotlib.pyplot as plt

# 从日志文件中读取数据
# log_filename = 'Mul-Datasets-Semantic-Segmentation/ltbgnn_5_datasets_joint_after_clip_model_km.log'

log_filename1 = 'city_seg_finetune_ori.log'
log_filename2 = 'city_seg_finetune_modify_cvcuda4.log'

y1 = []  # 存储第一个数组的数据
y2 = []  # 存储第二个数组的数据
y3 = []
y4 = []
y5 = []
y6 = []
y7 = []
y_mean = []
y_mean2 = []


# # 打开日志文件并读取数据
# with open('gnn_adv.log', 'r') as file:
#     for line in file:
#         if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
#             data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
#             # print(data)
#             data = [float(i) for i in data]  
#             y1.append(sum(data)/len(data))  
            
# with open('gnn_mse.log', 'r') as file:
#     for line in file:
#         if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
#             data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
#             # print(data)
#             data = [float(i) for i in data]  
#             y2.append(sum(data)/len(data))  

with open(log_filename1, 'r') as file:
    for line in file:
        if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
            data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
            # print(data)
            data = [float(i) for i in data]  
            y1.append(data[0])  
            # y2.append(data[1])  
            # y3.append(data[2])  
            # y4.append(data[3])  
            # y5.append(data[4])  
            # y6.append(data[5]) 
            # y7.append(data[6])
            # y_mean.append(sum(data) / len(data))
        elif line.find("loss: ") != -1:
            loss_index = line.find("loss: ")

            start_index = loss_index + len("loss: ")
            comma_index = line.find(",", start_index)

            if comma_index != -1:
                # 提取子字符串
                loss_value = float(line[start_index:comma_index].strip())
                y2.append(loss_value)
                # print("Loss值为:", loss_value)


with open(log_filename2, 'r') as file:
    for line in file:
        if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
            data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
            # print(data)
            data = [float(i) for i in data]  
            y3.append(data[0])  
            # y2.append(data[1])  
            # y3.append(data[2])  
            # y4.append(data[3])  
            # y5.append(data[4])  
            # y6.append(data[5]) 
            # y7.append(data[6])
            # y_mean.append(sum(data) / len(data))
        elif line.find("loss: ") != -1:
            loss_index = line.find("loss: ")

            start_index = loss_index + len("loss: ")
            comma_index = line.find(",", start_index)

            if comma_index != -1:
                # 提取子字符串
                loss_value = float(line[start_index:comma_index].strip())
                y4.append(loss_value)

# y5 = []
# y6 = []
# y7 = []
# y8 = []
# with open(log_filename2, 'r') as file:
#     for line in file:
#         if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
#             data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
#             # print(data)
#             data = [float(i) for i in data]  
#             y5.append(data[0])  
#             y6.append(data[1])  
#             y7.append(data[2])  
#             y8.append(sum(data) / len(data))
# 创建一个图形对象
plt.figure(figsize=(10, 8))
x = [(i) * 1e4 for i in range(1, 5)]
x2 = [(i) * 1e2 for i in range(1, 401)]

# 绘制折线图
# plt.plot(x, y1, label='CS', linestyle='-', marker='.')
# plt.plot(x, y2, label='MAPI', linestyle='-', marker='.')
# plt.plot(x, y3, label='SUN', linestyle='-', marker='.')
# plt.plot(x, y4, label='BDD', linestyle='-', marker='.')
# plt.plot(x, y5, label='IDD', linestyle='-', marker='.')
# plt.plot(x, y6, label='ADE', linestyle='-', marker='.')
# plt.plot(x, y7, label='COCO', linestyle='-', marker='.')
# plt.plot(x, y1[:4], label='OpenCV', linestyle='-', marker='.')
# plt.plot(x, y3, label='CVCUDA', linestyle='-', marker='.')

plt.plot(x2, y2[:400], label='OpenCV', linestyle='-', marker='.')
plt.plot(x2, y4[:400], label='CVCUDA', linestyle='-', marker='.')
# plt.plot(x, y5[:20], label='CS', linestyle='--', color='blue', marker='*')
# plt.plot(x, y6[:20], label='ADE', linestyle='--', color='red', marker='*')
# plt.plot(x, y7[:20], label='COCO', linestyle='--', color='green', marker='*')
# plt.plot(x, y8[:20], label='mean', linestyle='--', color='black', marker='*')
# plt.plot(x, y4, label='bdd', linestyle=':', color='purple')
# plt.plot(x, y5, label='idd', linestyle='-', color='black')
# plt.plot(x, y6, label='coco', linestyle='--', color='c')
# plt.plot(x, y7, label='idd', linestyle='-.', color='yellow')

# 添加标题和标签
# plt.title('Data from Log File')
plt.xlabel('Training iterations')
plt.ylabel('Loss')
# plt.axhline(0.1, color='black', linestyle='--')
# plt.axvline(100000, color='black', linestyle='--')
# plt.axvline(300000, color='blue', linestyle='--')
# plt.axvline(700000, color='black', linestyle='--')
# plt.axvline(900000, color='blue', linestyle='--')
# 添加图例
plt.legend()

# 保存图形到文件
plt.savefig('SEG_Training.png')
# plt.savefig('adv_loss.pdf')

# # 显示图形（可选）
# plt.show()
