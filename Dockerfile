# The Dockerfile for building the latest Dynamatic
# 
# Usage:
#
# 1. Creating a docker image:
# 
# ```bash 
# $ docker build -t dynamatic-image .
# ```
#
# 2. Running the docker image: 
#
# ```bash 
# $ docker run -it -u $(id -u):$(id -g) -v "$(pwd):/home/ubuntu/dynamatic" -w "/home/ubuntu/dynamatic" dynamatic-image /bin/bash
# ```
# 
# Remarks:
#   1. We run the container under the same UID and GID as your user is so that
#   any files created by the container in the host file system will be owned by
#   you. Take for instance this command, that creates a file called test.txt in
#   the current directory on the host
#
#   2. We mount the current directory inside the container: since the memory
#   state of the container is volatile, this guarantees that the changes we did
#   outside does not get lost.

# [START Installing the dependency]
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN \
  apt-get -y update && \
  apt-get -y upgrade && \
  apt-get install -y \
  --option APT::Immediate-Configure=false \
  sudo vim clang lld ccache cmake wget \
  ninja-build python3 openjdk-21-jdk \
  graphviz git curl gzip libreadline-dev \
  libboost-all-dev pkg-config python3-venv  \
  ghdl verilator
# [END Installing the dependency]

# The user does not need a password to run sudo
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ubuntu
