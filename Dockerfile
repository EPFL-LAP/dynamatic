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
# $ docker run -it -u dynamatic:dynamatic dynamatic-image /bin/bash
# ```
# 

# [START Installing the dependency]
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y upgrade
RUN \
  apt-get install -y \
  --option APT::Immediate-Configure=false \
  clang lld ccache cmake \
  ninja-build python3 openjdk-21-jdk \
  graphviz git curl gzip libreadline-dev \
  libboost-all-dev pkg-config coinor-cbc \
  coinor-libcbc-dev verilator ghdl
# [END Installing the dependency]

# [START Create a user called "dynamatic"]
# Add a user
RUN groupadd --gid 1000 dynamatic && \
  useradd --uid 1000 --gid dynamatic --shell /bin/bash --create-home dynamatic

USER dynamatic
ARG workdir="/home/dynamatic"
WORKDIR $workdir
# [END Create a user called "dynamatic"]

# [START Install SBT]
RUN \
    curl -fL "https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz" | \
    gzip -d > cs && chmod +x cs && ./cs setup -y && rm cs

ENV PATH="$PATH:$workdir/.local/share/coursier/bin"

RUN sbt --version
# [END Install SBT]

# [START Clone and build the latest Dynamatic]
RUN git clone \
  --recurse-submodules \
  "https://github.com/EPFL-LAP/dynamatic.git" "$workdir/dynamatic"

RUN cd "$workdir/dynamatic" && \
  bash "./build.sh" --release
# [END Clone and build the latest Dynamatic]
