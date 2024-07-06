# Use the official ROS Noetic desktop full image as a base
FROM osrf/ros:noetic-desktop-full

ENV http_proxy=http://localhost:3128
ENV https_proxy=http://localhost:3128
ENV ftp_proxy=http://localhost:3128

# Install required dependencies and tools
RUN apt-get update && apt-get install -y \
    tmux zsh git vim make cmake tree aptitude \
    htop curl rsync software-properties-common \
    locales mlocate aria2 python3-pip \
    libomp-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev \
    libboost-all-dev libglm-dev libglfw3-dev libpng-dev libjpeg-dev libeigen3-dev libfmt-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /root

# Install qhull
RUN git clone https://github.com/qhull/qhull.git && \
    cd qhull && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd /root && rm -rf qhull

# Install ceres-solver
RUN git clone https://github.com/ceres-solver/ceres-solver.git && \
    cd ceres-solver && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DUSE_CUDA=OFF && \
    make -j$(nproc) && \
    make install && \
    cd /root && rm -rf ceres-solver

# Install Iridescence for visualization
RUN git clone https://github.com/koide3/iridescence --recursive && \
    cd iridescence && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd /root && rm -rf iridescence

# Source ROS setup files and set CMAKE_PREFIX_PATH
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
ENV CMAKE_PREFIX_PATH /opt/ros/noetic:$CMAKE_PREFIX_PATH

# Check the installation paths
RUN echo "Checking library paths..." && \
    # ls /usr/local/lib && \
    # ls /usr/lib && \
    find /usr/local/lib -name "libiridescence*" && \
    find /usr/lib -name "libiridescence*" && \
    find /usr/local/lib -name "gl_imgui" && \
    find /usr/lib -name "gl_imgui"

# Copy the local PointCloudProcessor folder into the Docker image
COPY PointCloudProcessor /root/PointCloudProcessor

RUN cd /root/PointCloudProcessor && \
    rm -rf build && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) 

# Set environment variables
ENV HOME=/root
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=unix$DISPLAY

# X11 forwarding
VOLUME /tmp/.X11-unix:/tmp/.X11-unix:rw

# Set proxy environment variables (if needed)
# Note: Uncomment and set your proxy if required
# ENV http_proxy=http://localhost:3128
# ENV https_proxy=http://localhost:3128
# ENV ftp_proxy=http://localhost:3128

# Copy any additional scripts or configuration files if needed
# COPY some-script.sh /root/some-script.sh
# RUN chmod +x /root/some-script.sh

# Set the default shell to zsh (optional)
# RUN chsh -s /bin/zsh

# Set the entrypoint or default command if needed
# ENTRYPOINT ["/root/some-script.sh"]
# CMD ["bash"]

