FROM ubuntu:latest

RUN apt-get update && apt-get install -y cmake python3-distutils python3-dev python3-pip python3-pybind11 zlib1g-dev libboost-all-dev libeigen3-dev freeglut3-dev libgmp3-dev

RUN pip3 install pybind11 opencv-python matplotlib

COPY . /codim-ipc

WORKDIR /codim-ipc
