# 基础镜像选择适合你的环境的Python版本
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制conda环境文件和requirements.txt到容器中
COPY environment.yml .
COPY requirements.txt .

# # 创建并激活conda环境
# RUN conda env create -f environment.yml && \
#     conda activate myenv



# 安装PyTorch
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

RUN conda install scipy

# 安装Python依赖项
RUN pip install -r requirements.txt

# 克隆GitHub仓库并安装
# RUN git clone https://github.com/username/repository.git \
#    && pip install -e repository
RUN pip install git+https://github.com/openai/CLIP.git

# 复制PyTorch程序代码到容器中
COPY . .

# 设置容器启动命令
CMD ["python", "tools/train_ltbgnn_all_datasets.py"]

