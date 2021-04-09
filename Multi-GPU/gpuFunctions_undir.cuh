#ifndef GPUFUNCTIONS_UNDIR_CUH
#define GPUFUNCTIONS_UNDIR_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "all_structure_undir.cuh"
#include <math.h>
using namespace std;

#define THREADS_PER_BLOCK 1024 //we can change it


//returns number of 1 in mask
__device__ int BitCount(unsigned int mask)
{
	unsigned int uCount;
	uCount = mask
		- ((mask >> 1) & 033333333333)
		- ((mask >> 2) & 011111111111);
	return ((uCount + (uCount >> 3))
		& 030707070707) % 63;
}

__device__ int FirstZeroBit(int i)
{
	i = ~i;
	return BitCount((i & (-i)) - 1);
}

/*
* computeSC function creates a 32 bit mask, where ith bit is set if one of the adjacent vertex has color i
*/
__device__ int computeSC(ColWt* AdjListFull_device, int* AdjListTracker_device, int node, int* vertexcolor) {
	int saturationColorMask = 0;
	for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++) {
		if (AdjListFull_device[j].flag != -1)
		{
			saturationColorMask = saturationColorMask | int(exp2(double(vertexcolor[AdjListFull_device[j].col])));
		}
	}
	return saturationColorMask;
}

__global__ void printAdj(ColWt* AdjListFull_device, int* AdjListTracker_device, int nodes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < nodes; index++)
		{
			printf("\nAdj list for %d:\n", index);
			for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
				if (AdjListFull_device[j].flag != -1)
				{
					printf("%d:", AdjListFull_device[j].col);
				}
			}
		}
	}
}

__global__ void printMask(int* saturationColorMask, int* AdjListTracker_device, int nodes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < nodes; index++)
		{
			int SC_size;
			unsigned int s = saturationColorMask[index];
			SC_size = BitCount(s);
			printf("node: %d  SCMask:%d  SC_size:%d\n", index, s, SC_size);
		}
	}
}

__global__ void computeSCMask(ColWt* AdjListFull_device, int* AdjListTracker_device, int nodes, int* saturationColorMask, int* vertexcolor) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nodes; index += blockDim.x * gridDim.x)
	{
		saturationColorMask[index] = computeSC(AdjListFull_device, AdjListTracker_device, index, vertexcolor);

	}
}


__global__ void deleteEdgeFromAdj(changeEdge* allChange_Del_device, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int node_1 = allChange_Del_device[index].node1;
		int node_2 = allChange_Del_device[index].node2;
		//int edge_weight = allChange_Del_device[index].edge_wt;

		//mark the edge as deleted in Adjlist
		for (int j = AdjListTracker_device[node_2]; j < AdjListTracker_device[node_2 + 1]; j++) {
			if (AdjListFull_device[j].col == node_1) {
				AdjListFull_device[j].flag = -1; //flag set -1 to indicate deleted
				//printf("inside del inedge: %d %d %d \n", node_1, node_2, edge_weight);
			}

		}
		for (int j = AdjListTracker_device[node_1]; j < AdjListTracker_device[node_1 + 1]; j++) {
			if (AdjListFull_device[j].col == node_2) {
				AdjListFull_device[j].flag = -1;
				//printf("inside del outedge: %d %d %d \n", node_1, node_2, edge_weight);
			}
		}
	}
}


__global__ void deleteEdgeFromAdj_border(changeEdge* allChange_Del_cross_device, int totalChangeEdges_Del_cross, ColWt* AdjListFull_border_device, int* AdjListTracker_border_device) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del_cross; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int node_1 = allChange_Del_cross_device[index].node1; //This node is in this partition
		int node_2 = allChange_Del_cross_device[index].node2; //This node is a opv

		//mark the edge as deleted in Adjlist
		//for (int j = AdjListTracker_border_device[node_2]; j < AdjListTracker_border_device[node_2 + 1]; j++) {
		//	if (AdjListFull_border_device[j].col == node_1) {
		//		AdjListFull_border_device[j].flag = -1; //flag set -1 to indicate deleted
		//		//printf("inside del inedge: %d %d %d \n", node_1, node_2, edge_weight);
		//	}

		//}
		for (int j = AdjListTracker_border_device[node_1]; j < AdjListTracker_border_device[node_1 + 1]; j++) {
			if (AdjListFull_border_device[j].col == node_2) {
				AdjListFull_border_device[j].flag = -1;
			}
		}
	}
}



__global__ void deleteEdge(changeEdge* allChange_Del_device, int* vertexcolor, int* previosVertexcolor, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_del, int* change, int* borderVertexFlag, int* vertexcolor_opv, ColWt* AdjListFull_border_device, int* AdjListTracker_border_device, int* affected_marked_border) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int targeted_node = vertexcolor[allChange_Del_device[index].node1] > vertexcolor[allChange_Del_device[index].node2] ? allChange_Del_device[index].node1 : allChange_Del_device[index].node2;

		int SCMask = computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor);
		if (borderVertexFlag[targeted_node] == 1) //If node is a border vertex
		{
			int SCMask_opv = computeSC(AdjListFull_border_device, AdjListTracker_border_device, targeted_node, vertexcolor_opv);
			SCMask = SCMask | SCMask_opv;
		}
		int smallest_available_color = FirstZeroBit(SCMask);

		if (smallest_available_color != vertexcolor[targeted_node])
		{
			previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
			vertexcolor[targeted_node] = smallest_available_color;
			affected_del[targeted_node] = 1;
			*change = 1;
			if (borderVertexFlag[targeted_node] == 1) //If node is a border vertex
			{
				affected_marked_border[targeted_node] = 1;
			}
			//printf("color of %d became:%d", targeted_node, vertexcolor[targeted_node]);
		}
		//printf("affected_del flag for %d = %d \n *change = %d\n", targeted_node, affected_del[targeted_node], *change);
	}
}

__global__ void deleteEdge_cross(changeEdge* allChange_Del_cross_device, int* vertexcolor, int* previosVertexcolor, int* vertexcolor_opv, /*int* previosVertexcolor_opv,*/ int totalChangeEdges_Del_cross, ColWt* AdjListFull_device, int* AdjListTracker_device, ColWt* AdjListFull_border_device, int* AdjListTracker_border_device, int* affected_del, int* affected_marked_border, int* change) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del_cross; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int targeted_node = vertexcolor[allChange_Del_cross_device[index].node1] > vertexcolor[allChange_Del_cross_device[index].node2] ? allChange_Del_cross_device[index].node1 : allChange_Del_cross_device[index].node2;
		if (targeted_node == allChange_Del_cross_device[index].node1) //node2 is an opv. So we skip if targeted_node == allChange_Del_cross_device[index].node2
		{
			int SCMask_internal = computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor);
			int SCMask_opv = computeSC(AdjListFull_border_device, AdjListTracker_border_device, targeted_node, vertexcolor_opv);
			int SCMask = SCMask_internal | SCMask_opv;
			int smallest_available_color = FirstZeroBit(SCMask);

			if (smallest_available_color != vertexcolor[targeted_node])
			{
				previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
				vertexcolor[targeted_node] = smallest_available_color;
				affected_del[targeted_node] = 1;
				affected_marked_border[targeted_node] = 1;
				*change = 1;
				//printf("color of %d became:%d", targeted_node, vertexcolor[targeted_node]);
			}
		}
		//printf("affected_del flag for %d = %d \n *change = %d\n", targeted_node, affected_del[targeted_node], *change);
	}
}

__global__ void compute_SC_size(int total_borderVertex,int* borderVertexList, int* SC_size_border, ColWt* AdjListFull_device, int* AdjListTracker_device, ColWt* AdjListFull_border_device, int* AdjListTracker_border_device, int* vertexcolor, int* vertexcolor_opv)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_borderVertex; index += blockDim.x * gridDim.x)
	{
		int targeted_node = borderVertexList[index];
		unsigned int SCMask_internal = computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor);
		unsigned int SCMask_opv = computeSC(AdjListFull_border_device, AdjListTracker_border_device, targeted_node, vertexcolor_opv);
		unsigned int SCMask = SCMask_internal | SCMask_opv;
		int SC_size = BitCount(SCMask);
		SC_size_border[targeted_node] = SC_size;
	}
}

__global__ void insEdge(changeEdge* allChange_Ins_device, int* vertexcolor, int* previosVertexcolor, int totalChangeEdges_Ins, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* change, int* borderVertexFlag, int* vertexcolor_opv, ColWt* AdjListFull_border_device, int* AdjListTracker_border_device, int* affected_marked_border) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Ins; index += blockDim.x * gridDim.x)
	{
		if (vertexcolor[allChange_Ins_device[index].node1] == vertexcolor[allChange_Ins_device[index].node2])
		{
			int SC_size_node1;
			unsigned int sc_node1 = computeSC(AdjListFull_device, AdjListTracker_device, allChange_Ins_device[index].node1, vertexcolor);
			if (borderVertexFlag[allChange_Ins_device[index].node1] == 1) //If node is a border vertex
			{
				int SCMask_opv = computeSC(AdjListFull_border_device, AdjListTracker_border_device, allChange_Ins_device[index].node1, vertexcolor_opv);
				sc_node1 = sc_node1 | SCMask_opv;
			}
			SC_size_node1 = BitCount(sc_node1);
			int SC_size_node2;
			unsigned int sc_node2 = computeSC(AdjListFull_device, AdjListTracker_device, allChange_Ins_device[index].node2, vertexcolor);
			if (borderVertexFlag[allChange_Ins_device[index].node2] == 1) //If node is a border vertex
			{
				int SCMask_opv = computeSC(AdjListFull_border_device, AdjListTracker_border_device, allChange_Ins_device[index].node2, vertexcolor_opv);
				sc_node1 = sc_node1 | SCMask_opv;
			}
			SC_size_node2 = BitCount(sc_node2);
			int targeted_node = SC_size_node1 < SC_size_node2 ? allChange_Ins_device[index].node1 : allChange_Ins_device[index].node2;
			int SCMask = SC_size_node1 < SC_size_node2 ? sc_node1 : sc_node2;
			
			int smallest_available_color = FirstZeroBit(SCMask);
			previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
			vertexcolor[targeted_node] = smallest_available_color;
			affected_marked[targeted_node] = 1;
			if (borderVertexFlag[targeted_node] == 1) //If node is a border vertex
			{
				affected_marked_border[targeted_node] = 1;
			}
			*change = 1;
			//printf("Ins: color of %d became:%d", targeted_node, vertexcolor[targeted_node]);	
			//printf("affected_ins flag for %d = %d \n *change = %d\n", targeted_node, affected_marked[targeted_node], *change);
		}

	}
}

__global__ void insEdge_cross(changeEdge* allChange_Ins_cross_device, int* vertexcolor, int* previosVertexcolor, int* vertexcolor_opv, int totalChangeEdges_Ins_cross, ColWt* AdjListFull_device, int* AdjListTracker_device, ColWt* AdjListFull_border_device, int* AdjListTracker_border_device, int* affected_marked, int* affected_marked_border, int* change, int* SC_size_border, int* SC_size_opv) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Ins_cross; index += blockDim.x * gridDim.x)
	{
		if (vertexcolor[allChange_Ins_cross_device[index].node1] == vertexcolor_opv[allChange_Ins_cross_device[index].node2])
		{
			int SC_size_node1 = SC_size_border[allChange_Ins_cross_device[index].node1];
			int SC_size_node2 = SC_size_opv[allChange_Ins_cross_device[index].node2];
			int targeted_node = SC_size_node1 < SC_size_node2 ? allChange_Ins_cross_device[index].node1 : allChange_Ins_cross_device[index].node2;
			if (targeted_node == allChange_Ins_cross_device[index].node1)
			{
				int SCMask_internal = computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor);
				int SCMask_opv = computeSC(AdjListFull_border_device, AdjListTracker_border_device, targeted_node, vertexcolor_opv);
				int SCMask = SCMask_internal | SCMask_opv;
				int smallest_available_color = FirstZeroBit(SCMask);
				previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
				vertexcolor[targeted_node] = smallest_available_color;
				affected_marked[targeted_node] = 1;
				affected_marked_border[targeted_node] = 1;
				*change = 1;
			}
			
			//printf("Ins: color of %d became:%d", targeted_node, vertexcolor[targeted_node]);	
			//printf("affected_ins flag for %d = %d \n *change = %d\n", targeted_node, affected_marked[targeted_node], *change);
		}

	}
}

__global__ void findEligibleNeighbors(int* affectedNodeList, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* previosVertexcolor, int* vertexcolor, int* counter_del) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter_del; index += blockDim.x * gridDim.x)
	{
		int node = affectedNodeList[index];
		for (int j = AdjListTracker_device[node]; j < AdjListTracker_device[node + 1]; j++)
		{
			//Conflict resolution step
			if (AdjListFull_device[j].flag > -1 && vertexcolor[node] == vertexcolor[AdjListFull_device[j].col] && node < AdjListFull_device[j].col) //node < AdjListFull_device[j].col is used to select which vertex color to update in confusion
			{
				//atomicAdd(&AdjListFull_device[j].flag, -3);
				affected_marked[AdjListFull_device[j].col] = 1;
				//printf("AdjListFull_device[j].flag of %d = %d\n", AdjListFull_device[j].col, AdjListFull_device[j].flag);
			}
			//mark neighbor: when previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col] that means  node has directed edge toward j, 
			//or the node can affect the neighbor(this checks the total order condition)
			if (AdjListFull_device[j].flag > -1 && previosVertexcolor[node] < vertexcolor[AdjListFull_device[j].col])
			{
				affected_marked[AdjListFull_device[j].col] = 1;
			}

			//test for RMAT18 error check
			/*if (node == 3112 && AdjListFull_device[j].col == 108702)
			{
				printf("$$affected_marked[%d] = %d$$\n", AdjListFull_device[j].col, affected_marked[AdjListFull_device[j].col]);
				printf("$$vertexcolor[%d] = %d$$\n", node, vertexcolor[node]);
				printf("$$vertexcolor[%d] = %d$$\n", AdjListFull_device[j].col, vertexcolor[AdjListFull_device[j].col]);
			}*/


		}
	}
}

__global__ void recolorNeighbor(int* affectedNodeList, int* vertexcolor, int* previosVertexcolor, ColWt* AdjListFull_device, int* AdjListTracker_device, int* affected_marked, int* counter_del, int* change) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter_del; index += blockDim.x * gridDim.x)
	{
		int targeted_node = affectedNodeList[index];
		int SCMask = computeSC(AdjListFull_device, AdjListTracker_device, targeted_node, vertexcolor);
		int smallest_available_color = FirstZeroBit(SCMask);
		previosVertexcolor[targeted_node] = vertexcolor[targeted_node];
		vertexcolor[targeted_node] = smallest_available_color;
		affected_marked[targeted_node] = 1;
		*change = 1;
		//printf("color of %d became:%d", targeted_node, vertexcolor[targeted_node]);
		//printf("affected_marked flag for %d = %d \n *change = %d\n", targeted_node, affected_marked[targeted_node], *change);
	}
}

__global__ void validate(ColWt* AdjListFull_device, int* AdjListTracker_device, int nodes, int* vertexcolor) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {

		for (int index = 0; index < nodes; index++)
		{
			//printf("\nAdj list for %d:\n", index);
			for (int j = AdjListTracker_device[index]; j < AdjListTracker_device[index + 1]; j++) {
				if (AdjListFull_device[j].flag != -1 && vertexcolor[AdjListFull_device[j].col] == vertexcolor[index])
				{
					printf("##Error found: %d %d", index, AdjListFull_device[j].col);
				}
			}
		}
	}
}


//used for affected_del
struct predicate
{
	__host__ __device__
		bool operator()(int x)
	{
		return x == 1;
	}
};

//used for affected_all
struct predicate2
{
	__host__ __device__
		bool operator()(int x)
	{
		return x > 0;
	}
};


#endif