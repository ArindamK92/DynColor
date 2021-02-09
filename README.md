"# DynColor" 

1st arg: original graph file name
2nd arg: no. of nodes
3rd arg: no. of edges
4th arg: input colored graph file at time t_(n-1)
5th arg: change edges file name

Compile:
nvcc -o op Color_main.cu


Run:
Example:
C:\Users\Arindam\source\repos\DynamicColoring\DynamicColoring>op.exe graph.txt 10 19 prevColor.txt cE.txt
Reading input graph...
Reading input graph completed
Time taken to read input graph: 132 microseconds
Reading input changed edges data...
Reading input changed edges data completed. totalInsertion:9
Time taken to read input changed edges: 84 microseconds
creating 1D array from 2D to fit it in GPU
creating 1D array from 2D completed
Transferring graph data from CPU to GPU
**Time taken to transfer graph data from CPU to GPU: 3.494 milliseconds**
**Time taken for processing del edges: 0.133 milliseconds**
**Time taken for processing ins edges: 0.095 milliseconds**
**Time taken for processing affected neighbors: 5.841 milliseconds**
****Total Time for Vertex Color Update: 6.069 milliseconds****
