Step 1:Change our format graph to Metis format:
g++ -o op_toMetis CovertToMetisFormat.cpp
./op_toMetis graph.txt 4 > graph_metis


Step 2: partitioning (for more about partitioning using prev. partitions see github)
make pulp
OR>>> 
g++ -fopenmp -O3 -Wall -c pulp.cpp
g++ -fopenmp -O3 -Wall -o pulp pulp_main.cpp pulp.o

****The write_parts function in io.cpp is modified to generate multiple partition files containing vertex id for each partitions
file names are "partition<partition id>.txt"
Run on Foundry. Forge will not work due to lower gcc version. ****On desktop it does not work properly****
Use label propagation to get better partition(-l):
./pulp graphSample.txt [no. of partition] -l

(e.g. ./pulp roadNet-CA_Metis 4 -l)it creates 4 partition files and another file with all partitionID 


Step 3: Generate initial colors using Gunrock (single GPU)
to run and print only colors:
./test_color_11.0_x86_64_printcolor market bips98_606.mtx --undirected=1 --quiet > colorFile





How to run CUDA+MPI:
_____________________________
Let we need 4 GPU and 4 CPU: sinteractive -p cuda --time=04:00:00 --gres=gpu:4 --nodes=4 --mem=32000M
module load openmpi/4.0.3/gnu/9.2.0

nvcc -I/share/apps/common/openmpi/4.0.3/gnu/9.2.0/include -L/share/apps/common/openmpi/4.0.3/gnu/9.2.0/lib -lmpi kernel.cu -o op

mpirun -np 4 ./op roadNet-CA_wp_sor color_roadNet-CA_wp_sor roadNet-CA_cE_100K_50 > output.txt


**Note: How to get paths: do module avail->here all the items are actually a file(not dir)->cat the file->actual paths are written



CUDA+MPI+OpenMP:
_____________________________
Let we need 4 GPU and 4 CPU: 
sinteractive -p cuda --time=04:00:00 --gres=gpu:4 --nodes=4 --mem=32000M

module load openmpi/4.0.3/gnu/9.2.0

nvcc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt -gencode arch=compute_70,code=sm_70 -I/share/apps/common/openmpi/4.0.3/gnu/9.2.0/include -L/share/apps/common/openmpi/4.0.3/gnu/9.2.0/lib -lmpi kernel.cu -o op1

mpirun -np 4 ./op1 roadNet-CA_wp_sor color_roadNet-CA_wp_sor roadNet-CA_cE_100K_50 > output1.txt