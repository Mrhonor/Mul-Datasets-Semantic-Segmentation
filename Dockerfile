FROM nvidia/cuda:11.0-base

# 安装基本依赖
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx

# 设置工作目录
WORKDIR /app

# 复制conda环境文件和requirements.txt到容器中
COPY environment.yml .
COPY requirements.txt .

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 添加conda路径到环境变量
ENV PATH=/opt/conda/bin:$PATH

# 创建并激活conda环境
RUN conda env create -f environment.yml && \
    echo "source activate myenv" > ~/.bashrc
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# 安装PyTorch
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# 使用豆瓣源加速安装Python依赖项
RUN pip install -i https://pypi.douban.com/simple/ -U pip && \
    pip config set global.index-url https://pypi.douban.com/simple/

# 安装Python依赖项
RUN pip install -r requirements.txt

# 安装CLIP库
RUN pip install git+https://github.com/openai/CLIP.git

# 复制PyTorch程序代码到容器中
COPY . .

# 设置容器启动命令
CMD ["python", "tools/train_ltbgnn_all_datasets.py"]
