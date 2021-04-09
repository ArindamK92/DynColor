#ifndef SUPPORTING_H
#define SUPPORTING_H
#include <stdio.h>
#include <iostream>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <chrono>
#include <map>
#include "all_structure_undir.cuh"
#include "gpuFunctions_undir.cuh"

using namespace std;
//using namespace std::chrono;


#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }


void read_Partition_Vertices(map<int, int>& Global2LocalMap, map<int, int>& Local2GlobalMap, char* myfile, int * totalLocalVertices)
{
	FILE* graph_file;
	char line[128];
	int localID = 0;

	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int globalID;
		sscanf(line, "%d", &globalID);
		Global2LocalMap.insert(make_pair(globalID, localID));//It assigns a local vertex ID for all vertices in a partition
		Local2GlobalMap.insert(make_pair(localID, globalID));//This map helps to find GlobalID of a vertex. This convertion is required when sending some info to other partition
		localID++;
	}
	*totalLocalVertices = localID;
	fclose(graph_file);

	return;
}


void read_PartitionID_AllVertices(char* myfile, vector<int>& PartitionID_all)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int partID;
		sscanf(line, "%d", &partID);
		PartitionID_all.push_back(partID);
	}
	fclose(graph_file);

	return;
}

void transfer_data_to_GPU(vector<ColList>& AdjList, int*& AdjListTracker, vector<ColWt>& AdjListFull, ColWt*& AdjListFull_device,
	int totalLocalVertices,int totalLocalEdges, int*& AdjListTracker_device, bool zeroInsFlag,
	vector<changeEdge>& allChange_Ins, changeEdge*& allChange_Ins_device, int totalChangeEdges_Ins,
	int deviceId, int totalChangeEdges_Del, bool zeroDelFlag, changeEdge*& allChange_Del_device,
	int*& counter, int*& affected_marked, int*& affectedNodeList, int*& previosVertexcolor, /*int*& updatedAffectedNodeList_del, int*& updated_counter_del,*/ vector<changeEdge>& allChange_Del, size_t  numberOfBlocks)
{
	//cudaError_t cudaStatus;

	//create 1D array from 2D to fit it in GPU
	//cout << "creating 1D array from 2D to fit it in GPU" << endl;
	AdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < totalLocalVertices; i++) {
		AdjListTracker[i + 1] = AdjListTracker[i] + AdjList.at(i).size();
		AdjListFull.insert(std::end(AdjListFull), std::begin(AdjList.at(i)), std::end(AdjList.at(i)));
	}
	//cout << "creating 1D array from 2D completed" << endl;


	//Transferring input graph and change edges data to GPU
	//cout << "Transferring graph data from CPU to GPU" << endl;
	auto startTime_transfer = high_resolution_clock::now();

	CUDA_RT_CALL(cudaMallocManaged(&AdjListFull_device, totalLocalEdges * sizeof(ColWt))); //totalLocalEdges = (2 * (edges + totalInsertion))
	std::copy(AdjListFull.begin(), AdjListFull.end(), AdjListFull_device);


	CUDA_RT_CALL(cudaMalloc((void**)&AdjListTracker_device, (totalLocalVertices + 1) * sizeof(int)));
	CUDA_RT_CALL(cudaMemcpy(AdjListTracker_device, AdjListTracker, (totalLocalVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

	////Asynchronous prefetching of data
	CUDA_RT_CALL(cudaMemPrefetchAsync(AdjListFull_device, totalLocalEdges * sizeof(ColWt), deviceId));

	if (zeroInsFlag != true) {
		CUDA_RT_CALL(cudaMallocManaged(&allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge)));
		std::copy(allChange_Ins.begin(), allChange_Ins.end(), allChange_Ins_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		CUDA_RT_CALL(cudaMemAdvise(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId));
		//Asynchronous prefetching of data
		CUDA_RT_CALL(cudaMemPrefetchAsync(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), deviceId));
	}

	if (zeroDelFlag != true) {
		CUDA_RT_CALL(cudaMallocManaged(&allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge)));
		std::copy(allChange_Del.begin(), allChange_Del.end(), allChange_Del_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		CUDA_RT_CALL(cudaMemAdvise(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId));
		//Asynchronous prefetching of data
		CUDA_RT_CALL(cudaMemPrefetchAsync(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), deviceId));

		counter = 0;
		CUDA_RT_CALL(cudaMallocManaged(&counter, sizeof(int)));
		CUDA_RT_CALL(cudaMallocManaged(&affected_marked, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMemset(affected_marked, 0, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMallocManaged(&affectedNodeList, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMemset(affectedNodeList, 0, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMallocManaged(&previosVertexcolor, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMemset(previosVertexcolor, -1, totalLocalVertices * sizeof(int)));
		/*cudaMallocManaged(&updatedAffectedNodeList_del, totalLocalVertices * sizeof(int));*/
		/*updated_counter_del = 0;
		cudaMallocManaged(&updated_counter_del, sizeof(int));*/

		//modify adjacency list to adapt the deleted edges
		deleteEdgeFromAdj << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device);
		CUDA_RT_CALL(cudaGetLastError());
		CUDA_RT_CALL(cudaDeviceSynchronize());
	}

	auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
	auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
	cout << "**Time taken to transfer graph data from CPU to GPU: "
		<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;
}


//this function is related to border vertices, opv, and cross edges 
void transfer_border_data_to_GPU(vector<ColList>& AdjList_border, int*& AdjListTracker_border, vector<ColWt>& AdjListFull_border, ColWt*& AdjListFull_border_device,
	int totalLocalVertices, int totalCrossEdges, int*& AdjListTracker_border_device, bool zeroInsFlag_cross,
	vector<changeEdge>& allChange_Ins_cross, changeEdge*& allChange_Ins_cross_device, int totalChangeEdges_Ins_cross,
	int deviceId, int totalChangeEdges_Del_cross, bool zeroDelFlag_cross, changeEdge*& allChange_Del_cross_device,
	int*& counter_border, int*& affected_marked_border, int*& affectedNodeList_border, int*& previosVertexcolor_opv, /*int*& updatedAffectedNodeList_del, int*& updated_counter_del,*/ vector<changeEdge>& allChange_Del_cross, size_t  numberOfBlocks, int total_opv)
{
	//cudaError_t cudaStatus;

	//create 1D array from 2D to fit it in GPU
	//cout << "creating 1D array from 2D to fit it in GPU" << endl;
	AdjListTracker_border[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < totalLocalVertices; i++) {
		AdjListTracker_border[i + 1] = AdjListTracker_border[i] + AdjList_border.at(i).size();
		AdjListFull_border.insert(std::end(AdjListFull_border), std::begin(AdjList_border.at(i)), std::end(AdjList_border.at(i)));
	}
	//cout << "creating 1D array from 2D completed" << endl;


	//Transferring input graph and change edges data to GPU
	//cout << "Transferring graph data from CPU to GPU" << endl;
	auto startTime_transfer = high_resolution_clock::now();

	CUDA_RT_CALL(cudaMallocManaged(&AdjListFull_border_device, totalCrossEdges * sizeof(ColWt))); //totalCrossEdges includes inserted new cross edges also
	std::copy(AdjListFull_border.begin(), AdjListFull_border.end(), AdjListFull_border_device);


	CUDA_RT_CALL(cudaMalloc((void**)&AdjListTracker_border_device, (totalLocalVertices + 1) * sizeof(int)));
	CUDA_RT_CALL(cudaMemcpy(AdjListTracker_border_device, AdjListTracker_border, (totalLocalVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

	////Asynchronous prefetching of data
	CUDA_RT_CALL(cudaMemPrefetchAsync(AdjListFull_border_device, totalCrossEdges * sizeof(ColWt), deviceId));

	if (zeroInsFlag_cross != true) {
		CUDA_RT_CALL(cudaMallocManaged(&allChange_Ins_cross_device, totalChangeEdges_Ins_cross * sizeof(changeEdge)));
		std::copy(allChange_Ins_cross.begin(), allChange_Ins_cross.end(), allChange_Ins_cross_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		CUDA_RT_CALL(cudaMemAdvise(allChange_Ins_cross_device, totalChangeEdges_Ins_cross * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId));
		//Asynchronous prefetching of data
		CUDA_RT_CALL(cudaMemPrefetchAsync(allChange_Ins_cross_device, totalChangeEdges_Ins_cross * sizeof(changeEdge), deviceId));
	}

	if (zeroDelFlag_cross != true) {
		CUDA_RT_CALL(cudaMallocManaged(&allChange_Del_cross_device, totalChangeEdges_Del_cross * sizeof(changeEdge)));
		std::copy(allChange_Del_cross.begin(), allChange_Del_cross.end(), allChange_Del_cross_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		CUDA_RT_CALL(cudaMemAdvise(allChange_Del_cross_device, totalChangeEdges_Del_cross * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId));
		//Asynchronous prefetching of data
		CUDA_RT_CALL(cudaMemPrefetchAsync(allChange_Del_cross_device, totalChangeEdges_Del_cross * sizeof(changeEdge), deviceId));

		counter_border = 0;
		CUDA_RT_CALL(cudaMallocManaged(&counter_border, sizeof(int)));
		CUDA_RT_CALL(cudaMallocManaged(&affected_marked_border, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMemset(affected_marked_border, 0, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMallocManaged(&affectedNodeList_border, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMemset(affectedNodeList_border, 0, totalLocalVertices * sizeof(int)));
		CUDA_RT_CALL(cudaMallocManaged(&previosVertexcolor_opv, total_opv * sizeof(int))); //total_opv is used here. previosVertexcolor_opv stores prev color of an opv
		CUDA_RT_CALL(cudaMemset(previosVertexcolor_opv, -1, total_opv * sizeof(int)));

		//modify adjacency list to adapt the deleted edges
		deleteEdgeFromAdj_border << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_cross_device, totalChangeEdges_Del_cross, AdjListFull_border_device, AdjListTracker_border_device);
		CUDA_RT_CALL(cudaGetLastError());
		CUDA_RT_CALL(cudaDeviceSynchronize());
	}

	auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
	auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
	cout << "**Time taken to transfer graph data(border) from CPU to GPU: "
		<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;
}

#endif
