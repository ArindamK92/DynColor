#ifndef SUPPORTING_CU
#define SUPPORTING_CU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "all_structure_undir.cuh"
#include "gpuFunctions_undir.cuh"
//#include "bfs.cu"
using namespace std;
using namespace std::chrono;


void transfer_data_to_GPU(vector<ColList>& AdjList, int*& AdjListTracker, vector<ColWt>& AdjListFull, ColWt*& AdjListFull_device,
	int nodes, int edges, int totalInsertion, int*& AdjListTracker_device, 
	vector<changeEdge>& allChange, changeEdge*& allChange_device, int totalChangeEdges_Ins,
	int deviceId, int totalChangeEdges_Del, bool zeroDelFlag, changeEdge*& allChange_Del_device,
	int*& counter, int*& affected_marked, int*& affectedNodeList, int*& previosVertexcolor, /*int*& updatedAffectedNodeList_del, int*& updated_counter_del,*/ vector<changeEdge>& allChange_Del, size_t  numberOfBlocks)
{
	cudaError_t cudaStatus;

	//create 1D array from 2D to fit it in GPU
	cout << "creating 1D array from 2D to fit it in GPU" << endl;
	AdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < nodes; i++) {
		AdjListTracker[i + 1] = AdjListTracker[i] + AdjList.at(i).size();
		AdjListFull.insert(std::end(AdjListFull), std::begin(AdjList.at(i)), std::end(AdjList.at(i)));
	}
	cout << "creating 1D array from 2D completed" << endl;


	//Transferring input graph and change edges data to GPU
	cout << "Transferring graph data from CPU to GPU" << endl;
	auto startTime_transfer = high_resolution_clock::now();

	cudaStatus = cudaMallocManaged(&AdjListFull_device, (2 * (edges + totalInsertion)) * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	std::copy(AdjListFull.begin(), AdjListFull.end(), AdjListFull_device);


	cudaStatus = cudaMalloc((void**)&AdjListTracker_device, (nodes + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListTracker_device");
	}
	cudaMemcpy(AdjListTracker_device, AdjListTracker, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(AdjListFull_device, edges * sizeof(ColWt), deviceId);


	cudaStatus = cudaMallocManaged(&allChange_device, totalChangeEdges_Ins * sizeof(changeEdge));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at allChange_Ins structure");
	}
	std::copy(allChange.begin(), allChange.end(), allChange_device);
	//set cudaMemAdviseSetReadMostly by the GPU for change edge data
	cudaMemAdvise(allChange_device, totalChangeEdges_Ins * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);
	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(allChange_device, totalChangeEdges_Ins * sizeof(changeEdge), deviceId);


	if (zeroDelFlag != true) {
		cudaStatus = cudaMallocManaged(&allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed at allChange_Del structure");
		}
		std::copy(allChange_Del.begin(), allChange_Del.end(), allChange_Del_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		cudaMemAdvise(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);
		//Asynchronous prefetching of data
		cudaMemPrefetchAsync(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), deviceId);

		counter = 0;
		cudaMallocManaged(&counter, sizeof(int));
		cudaMallocManaged(&affected_marked, nodes * sizeof(int));
		cudaMemset(affected_marked, 0, nodes * sizeof(int));
		cudaMallocManaged(&affectedNodeList, nodes * sizeof(int));
		cudaMemset(affectedNodeList, 0, nodes * sizeof(int));
		cudaMallocManaged(&previosVertexcolor, nodes * sizeof(int));
		cudaMemset(previosVertexcolor, -1, nodes * sizeof(int));
		/*cudaMallocManaged(&updatedAffectedNodeList_del, nodes * sizeof(int));*/
		/*updated_counter_del = 0;
		cudaMallocManaged(&updated_counter_del, sizeof(int));*/

		//modify adjacency list to adapt the deleted edges
		deleteEdgeFromAdj << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device);
		cudaDeviceSynchronize();
	}

	auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
	auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
	cout << "**Time taken to transfer graph data from CPU to GPU: "
		<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;
}
//
//void read_and_transfer_input_SSSPtree_to_GPU(char* inputSSSPfile, vector<ColList>& SSSPTreeAdjList, int*& SSSPTreeAdjListTracker, vector<int>& SSSPTreeAdjListFull,
//	RT_Vertex*& SSSP, int nodes, int edges, int*& SSSPTreeAdjListFull_device, int*& SSSPTreeAdjListTracker_device, vector<int>& hop, int deviceId, int*& d_hop)
//{
//	cudaError_t cudaStatus;
//
//	SSSPTreeAdjList.resize(nodes);
//	SSSPTreeAdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
//
//	cudaStatus = cudaMallocManaged(&SSSP, nodes * sizeof(RT_Vertex));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed at SSSP structure");
//	}
//	cout << "Reading input SSSP tree data..." << endl;
//	auto readSSSPstartTime = high_resolution_clock::now();//Time calculation starts
//	//read_SSSP(SSSP, inputSSSPfile, &nodes, SSSPTreeAdjList);
//
//
//	//New addition
//	SSSPTreeAdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
//	for (int i = 0; i < nodes; i++) {
//		SSSPTreeAdjListTracker[i + 1] = SSSPTreeAdjListTracker[i] + SSSPTreeAdjList.at(i).size();
//		SSSPTreeAdjListFull.insert(std::end(SSSPTreeAdjListFull), std::begin(SSSPTreeAdjList.at(i)), std::end(SSSPTreeAdjList.at(i)));
//	}
//
//
//	//Transferring SSSP tree data to GPU
//
//	cudaStatus = cudaMallocManaged(&SSSPTreeAdjListFull_device, (nodes - 1) * sizeof(int)); //SSSP tree has n-1 edges and we consider each edge 1 time
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed at SSSPTreeAdjListFull_device structure");
//	}
//	std::copy(SSSPTreeAdjListFull.begin(), SSSPTreeAdjListFull.end(), SSSPTreeAdjListFull_device);
//
//
//	cudaStatus = cudaMalloc((void**)&SSSPTreeAdjListTracker_device, (nodes + 1) * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed at SSSPTreeAdjListTracker_device");
//	}
//	cudaMemcpy(SSSPTreeAdjListTracker_device, SSSPTreeAdjListTracker, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
//
//
//	//compute hop
//	vector<bool> visited;
//	int startVertex = 0; //0 is considered as root vertex
//	visited = vector<bool>(nodes);
//	//hop = vector<int>(nodes);
//
//	const int size = nodes * sizeof(int);
//	cudaMalloc((void**)&d_hop, size);
//	hop = vector<int>(nodes, INT_MAX);
//	hop[startVertex] = 0;
//	cudaMemcpy(d_hop, hop.data(), size, cudaMemcpyHostToDevice);
//
//	auto startTime = chrono::steady_clock::now();
//	//bfsGPU(startVertex, nodes, SSSPTreeAdjListFull_device, SSSPTreeAdjListTracker_device, d_hop, visited);
//	auto endTime = std::chrono::steady_clock::now();
//	long duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
//	printf("Elapsed time for hop computation : %li ms.\n", duration);
//
//	//cudaDeviceSynchronize();
//	cudaMemcpy(&hop[0], d_hop, size, cudaMemcpyDeviceToHost);
//	//printHop(hop);
//
//	auto readSSSPstopTime = high_resolution_clock::now();//Time calculation ends
//	auto readSSSPduration = duration_cast<microseconds>(readSSSPstopTime - readSSSPstartTime);// duration calculation
//	cout << "Reading input SSSP tree data completed" << endl;
//	cout << "Time taken to read input input SSSP tree: " << readSSSPduration.count() << " microseconds" << endl;
//	//set cudaMemAdviseSetPreferredLocation at GPU for SSSP data
//	cudaMemAdvise(SSSP, nodes * sizeof(RT_Vertex), cudaMemAdviseSetPreferredLocation, deviceId);
//
//	//Asynchronous prefetching of data
//	cudaMemPrefetchAsync(SSSP, nodes * sizeof(RT_Vertex), deviceId);
//
//}
#endif