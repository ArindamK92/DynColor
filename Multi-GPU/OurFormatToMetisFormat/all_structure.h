#ifndef ALL_STRUCTURE_H
#define ALL_STRUCTURE_H
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
//#include <chrono>

using namespace std;
//using namespace std::chrono;


//#include <omp.h>









/******* Network Structures *********/
struct ColWt {
	int col;
	//int flag; //default 0, deleted -1
};

//Structure for Edge
struct Edge
{
	int node1;
	int node2;
	double edge_wt;
};



struct changeEdge {
	int node1;
	int node2;
	int inst;
};

typedef vector<ColWt> ColList;




//
//
//// Data Structure for each vertex in the rooted tree
//struct RT_Vertex
//{
//	int Parent; //mark the parent in the tree
//	int EDGwt; //mark weight of the edge
//	int Dist;  //Distance from root
//	int Update;  //Whether the distance of this edge was updated / affected
//
//
//
//};
//
//
//////functions////
////Node starts from 0
//
//
//
//
///*
// readin_changes function reads the change edges
// Format of change edges file: node1 node2 edge_weight insert_status
// insert_status = 1 for insertion. insert_status = 0 for deletion.
// */
//void readin_changes(char* myfile, vector<changeEdge>& allChange_Ins, vector<changeEdge>& allChange_Del, vector<ColList>& AdjList, int& totalInsertion, int* vertexcolor)
//{
//	cout << "Reading input changed edges data..." << endl;
//	auto readCEstartTime = high_resolution_clock::now();//Time calculation starts
//	FILE* graph_file;
//	char line[128];
//	graph_file = fopen(myfile, "r");
//	while (fgets(line, 128, graph_file) != NULL)
//	{
//		int n1, n2, wt, inst_status;
//		changeEdge cE;
//		sscanf(line, "%d %d %d %d", &n1, &n2, &wt, &inst_status); //edge wt is there in input file. But we don't need it. So we will ignore it.
//		cE.node1 = n1;
//		cE.node2 = n2;
//		cE.inst = inst_status;
//
//
//		//add change edges with inst status = 1 to Adjlist
//		if (inst_status == 1)
//		{
//			ColWt c1, c2;
//			c1.col = n1;
//			c1.flag = 0;
//			c2.col = n2;
//			c2.flag = 0;
//			totalInsertion++;
//			AdjList.at(n2).push_back(c1);
//			AdjList.at(n1).push_back(c2);
//			if (vertexcolor[n1] == vertexcolor[n2]) //optimization
//			{
//				allChange_Ins.push_back(cE);
//			}
//
//		}
//		else if (inst_status == 0) {
//			allChange_Del.push_back(cE);
//		}
//
//	}
//	fclose(graph_file);
//	auto readCEstopTime = high_resolution_clock::now();//Time calculation ends
//	auto readCEduration = duration_cast<microseconds>(readCEstopTime - readCEstartTime);// duration calculation
//	cout << "Reading input changed edges data completed. totalInsertion:" << totalInsertion << endl;
//	cout << "Time taken to read input changed edges: " << readCEduration.count() << " microseconds" << endl;
//	return;
//}
//
//
///*
//read_Input_Color reads input color lebel file.
//accepted data format: node color
//*/
//void read_Input_Color(int* vertexcolor, char* myfile)
//{
//	FILE* graph_file;
//	char line[128];
//
//	graph_file = fopen(myfile, "r");
//	while (fgets(line, 128, graph_file) != NULL)
//	{
//		int node, color;
//		sscanf(line, "%d %d", &node, &color);
//		vertexcolor[node] = color;
//	}
//	fclose(graph_file);
//
//	return;
//}

/*
read_graphEdges reads the original graph file
accepted data format: node1 node2 edge_weight
we consider only undirected graph here. for edge e(a,b) with weight W represented as : a b W
*/
int read_graphEdges(vector<ColList>& AdjList, char* myfile)
{
	/*cout << "Reading input graph..." << endl;*/
	//auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	int edges = 0;
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2, wt;
		sscanf(line, "%d %d %d", &n1, &n2, &wt); //our input graph has wt. But we don't need it. So we will ignore wt.
		ColWt c1, c2;
		c1.col = n1;
		//c1.flag = 0;
		c2.col = n2;
		//c2.flag = 0;
		AdjList.at(n2).push_back(c1);
		AdjList.at(n1).push_back(c2);
		edges++;
	}
	fclose(graph_file);
	//auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	//auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	/*cout << "Reading input graph completed" << endl;*/
	/*cout << "Time taken to read input graph: " << readGraphduration.count() << " microseconds" << endl;*/
	return edges;
}



#endif
