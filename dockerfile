FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set environment variables for MuJoCo
ENV MUJOCO_VERSION=210 \
    MUJOCO_DIR=/root/.mujoco \
    LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH} \
    PATH=$LD_LIBRARY_PATH:$PATH \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:/usr/lib/x86_64-linux-gnu/libGLEW.so

# Install base dependencies
RUN apt-get update && apt-get install -y \
    sudo wget git \
    libglew-dev libgl-dev \
    qt5-default libxcb-xinerama0-dev \
    python3-dev build-essential libssl-dev libffi-dev libxml2-dev \
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 libopengl0 \
    libxslt1-dev zlib1g-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyQt5
RUN pip install --no-cache-dir PyQt5==5.14.2

# Install for data processing and visualization
RUN pip install pandas matplotlib seaborn

# Install MuJoCo
RUN mkdir -p ${MUJOCO_DIR} \
    && wget https://mujoco.org/download/mujoco${MUJOCO_VERSION}-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz \
    && tar -xvzf /tmp/mujoco.tar.gz -C ${MUJOCO_DIR} \
    && rm /tmp/mujoco.tar.gz

# Clone and install mujoco-py
RUN git clone https://github.com/openai/mujoco-py /opt/mujoco-py \
    && cd /opt/mujoco-py \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements.dev.txt \
    && pip install -e . --no-cache-dir

# Set working directory (your mounted workspace)
RUN pip install mujoco==2.3.5
RUN pip install gym==0.25.2
RUN pip install dm_control==1.0.12
RUN pip install git+https://github.com/denisyarats/dmc2gym.git
RUN pip install tensorboard termcolor pybullet scikit-image
RUN pip install hydra-core==1.0.4
RUN pip install transformers==4.32.0
RUN pip install "cython<3"
# Human preference labeling (train_PEBBLE_*_human.py): video I/O + frame resize
RUN pip install --no-cache-dir imageio imageio-ffmpeg opencv-python-headless

# Install Metaworld
RUN wget https://github.com/Farama-Foundation/Metaworld/archive/refs/tags/v2.0.0.tar.gz \
    && tar -xvzf v2.0.0.tar.gz \
    && cd Metaworld-2.0.0 \
    && pip install .
