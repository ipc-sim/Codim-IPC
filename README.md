# Source Codes for Codimensional Incremental Potential Contact (C-IPC)

## Reference

This repository provides source code for: 

Minchen Li, Danny M. Kaufman, Chenfanfu Jiang, [Codimensional Incremental Potential Contact](https://ipc-sim.github.io/C-IPC/), ACM Transactions on Graphics (SIGGRAPH  2021)

## Installation

### UBUNTU
Install python
```
sudo apt install python3-distutils python3-dev python3-pip python3-pybind11 zlib1g-dev libboost-all-dev libeigen3-dev freeglut3-dev libgmp3-dev
pip3 install pybind11
```
Build library
```
python build.py
```

### MacOS
Build library
```
python build_Mac.py
```

## To run paper examples

All examples can be executed with:
```
cd Projects/FEMShell
python batch.py
```
