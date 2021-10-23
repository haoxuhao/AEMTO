![](figures/AEMTO.png)
# Evolutionary Multi-task Optimization with Adaptive Knowledge Transfer
Source code for AEMTO.

# Get started
The code is only tested on linux OS. If you want to run on other OS, some equivalent environments should be set properly.

## Envs
The following software environments are required. 
```
cmake >= 3.2
g++ >= 6.4.0; use c++11 standard
python >= 3.0
```

## Build and Run with default arguments
```
git clone https://github.com/haoxuhao/AEMTO.git
cd AEMTO
pip install -r requirements.txt
mkdir build && cd build && cmake .. && make -j
cd ../bin
./AEMTO # && ./MATDE && ./SBO && ./MFEA
```
The detailed results are recorded in `Results/*/*.json`. 

## Run with specified arguments
File `./run.sh` lists some example params.
