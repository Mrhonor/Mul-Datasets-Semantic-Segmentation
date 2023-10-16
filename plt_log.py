import matplotlib.pyplot as plt

# 从日志文件中读取数据
log_filename = 'Mul-Datasets-Semantic-Segmentation/ltbgnn_5_datasets_joint_after_clip_model_km.log'


y1 = []  # 存储第一个数组的数据
y2 = []  # 存储第二个数组的数据
y3 = []
# y4 = []
# y5 = []
# y6 = []


# 打开日志文件并读取数据
with open('gnn_adv.log', 'r') as file:
    for line in file:
        if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
            data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
            # print(data)
            data = [float(i) for i in data]  
            y1.append(sum(data)/len(data))  
            
with open('gnn_mse.log', 'r') as file:
    for line in file:
        if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
            data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
            # print(data)
            data = [float(i) for i in data]  
            y2.append(sum(data)/len(data))  

with open('gnn_none.log', 'r') as file:
    for line in file:
        if line.startswith("|") and line.find('.') != -1:  # 仅处理以“|”开头的行
            data = line.strip("|").replace("|", '').strip().split()  # 去除“|”，然后用空格分隔数据
            # print(data)
            data = [float(i) for i in data]  
            y3.append(sum(data)/len(data))  

# 创建一个图形对象
plt.figure(figsize=(8, 6))
x = [(i) * 5e3 for i in range(1, 11)]
# 绘制折线图
plt.plot(x, y1[:10], label='Adv loss', linestyle='-', color='blue', marker='*')
plt.plot(x, y2[:10], label='MSE loss', linestyle='-', color='red', marker='*')
plt.plot(x, y3[:10], label='None', linestyle='-', color='green', marker='*')
# plt.plot(x, y4, label='bdd', linestyle=':', color='purple')
# plt.plot(x, y5, label='idd', linestyle='-', color='black')
# plt.plot(x, y6, label='coco', linestyle='--', color='c')
# plt.plot(x, y7, label='idd', linestyle='-.', color='yellow')

# 添加标题和标签
# plt.title('Data from Log File')
plt.xlabel('Training iterations')
plt.ylabel('Average mIoU')

# 添加图例
plt.legend()

# 保存图形到文件
plt.savefig('adv_loss.png')
plt.savefig('adv_loss.pdf')

# # 显示图形（可选）
# plt.show()
