#!/bin/bash

set -euxo pipefail

mkdir src
mkdir -p /home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models/standard
mkdir -p /home/clemens/xprun/queue

apt-get update
apt-get install --yes gnupg curl software-properties-common htop git rsync vim g++ 

#add-apt-repository ppa:graphics-drivers --yes
#apt-get update
#apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
#bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
#apt-get update
#apt-get install --yes cuda-10-1 libcudnn7

echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
apt-get update
apt-get install --yes openjdk-8-jdk sbt=0.13.16

pip install torch-scatter==2.0.5+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

cd src
git clone https://github.com/cswinter/CodeCraftGame.git
git checkout deepcodecraft
cd CodeCraftGame
sbt publishLocal

