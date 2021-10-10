# Install requirements

## openmpi

```
cd /opt/
wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.1.tar.gz
tar -xvf openmpi-3.0.1.tar.gz
./configure --prefix=/usr/local/openmpi

make

# MPI install to /usr/local/lib
sudo make install

sudo gedit /etc/profile
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

mpirun
#--------------------------------------------------------------------------
#mpirun could not find anything to do.
#
#It is possible that you forgot to specify how many processes to run
#via the "-np" argument.
#--------------------------------------------------------------------------
sudo make uninstall

```

## CUDA 
Refer to [NVIDIA](https://developer.nvidia.com/cuda-downloads). 


