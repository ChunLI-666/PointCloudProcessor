## Pull image

```bash
# docker pull koide3/direct_visual_lidar_calibration:noetic
docker pull osrf/ros:noetic-desktop-full
```

## Run docker image
```bash
docker run -it \
    --workdir=/sandbox/ \
    -v $HOME:/sandbox/ \
    -e HOME=/root/ \
    -e "QT_X11_NO_MITSHM=1" \
    -e GDK_SCALE \
    -e GDK_DPI_SCALE \
    -e DISPLAY=unix$DISPLAY \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name=pcd-colorization \
    --privileged \
    --network host \
    -v /etc/timezone:/etc/timezone:ro \
    -d osrf/ros:noetic-desktop-full 

# if GPU resource is avaiable, enable gpu support can accelerate the manual initial-guess
docker run -it \
    --gpus all \
    --workdir=/sandbox/ \
    -v $HOME:/sandbox/ \
    -e HOME=/root/ \
    -e "QT_X11_NO_MITSHM=1" \
    -e GDK_SCALE \
    -e GDK_DPI_SCALE \
    -e DISPLAY=unix$DISPLAY \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name=pcd-colorization \
    --privileged \
    --network host \
    -v /etc/timezone:/etc/timezone:ro \
    -d osrf/ros:noetic-desktop-full 
```


# attach to ros2 container
```bash

xhost +local:docker
docker attach pcd-colorization
```

## 2. Inside docker image


### 2.1. basic setup
```bash
# (for bosch PC only) setup proxy (put them in .bashrc or .zshrc)
export http_proxy=http://localhost:3128
export https_proxy=http://localhost:3128
export ftp_proxy=http://localhost:3128

# For personal pc via using proxy tools such as clash 
export http_proxy=http://localhost:7890
export https_proxy=http://localhost:7890
export ftp_proxy=http://localhost:7890
# test X11 forwarding
rviz

# install some tools, `-E` for proxy envs
sudo -E apt update
sudo -E apt install -y tmux zsh git vim make cmake tree aptitude \
                       htop curl rsync software-properties-common \
                       locales mlocate aria2 python3-pip \
                       libomp-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev \
                       libboost-all-dev libglm-dev libglfw3-dev libpng-dev libjpeg-dev libfmt-dev 



# install qhull
cd /root \
  && git clone https://github.com/qhull/qhull.git \
  && cd qhull/build \
  && cmake .. \
  && make -j$(nproc) \
  && make install \
  && rm -rf /root/qhull

# install ceres
cd /root \
  && git clone https://github.com/ceres-solver/ceres-solver \
  && mkdir ceres-solver/build \
  && cd ceres-solver/build \
  && cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DUSE_CUDA=OFF \
  && make -j$(nproc) \
  && make install \
  && rm -rf /root/ceres-solver

# Install Iridescence for visualization
cd /root \
  && git clone https://github.com/koide3/iridescence --recursive \
  && mkdir iridescence/build && cd iridescence/build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
  && make -j$(nproc) \
  && make install \
  && rm -rf /root/iridescence
```

### 2.2 Build and install
```bash
cd /sandbox && \
git clone --recursice https://github.com/ChunLI-666PointCloudProcessor.git &&\
cd PointCloudProcessor && \
mkdir build && cd build && \
cmake .. && make -j$(nproc)
``` 