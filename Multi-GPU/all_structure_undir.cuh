#ifndef ALL_STRUCTURE_UNDIR_CUH
#define ALL_STRUCTURE_UNDIR_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <chrono>
#include <map>
using namespace std;
using namespace std::chrono;


#include <omp.h>









/******* Network Structures *********/
struct ColWt {
	int col;
	int flag; //default 0, deleted -1
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






// Data Structure for each vertex in the rooted tree
struct RT_Vertex
{
	int Parent; //mark the parent in the tree
	int EDGwt; //mark weight of the edge
	int Dist;  //Distance from root
	int Update;  //Whether the distance of this edge was updated / affected



};


////functions////
//Node starts from 0




/*
 readin_changes function reads the change edges
 Format of change edges file: node1 node2 edge_weight insert_status
 insert_status = 1 for insertion. insert_status = 0 for deletion.
 */
void readin_changes(char* myfile, vector<changeEdge>& allChange_Ins, vector<changeEdge>& allChange_Del, vector<changeEdge>& allChange_Ins_cross, vector<changeEdge>& allChange_Del_cross, vector<ColList>& AdjList, /*int* vertexcolor,*/ map<int, int>& Global2LocalMap, /*set<int>& other_part_ngbr,*/ int*& borderVertexFlag, int* totalLocalEdges, int* totalCrossEdges, vector<ColList>& AdjList_border, map<int, int>& Global2LocalMap_opv, map<int, int>& Local2GlobalMap_opv, int* total_opv, vector<int>& PartitionID_all, set<int>*& bvPerPart)
{
	cout << "Reading input changed edges data..." << endl;
	auto readCEstartTime = high_resolution_clock::now();//Time calculation starts
	int totalCrossEdges_count = *totalCrossEdges;
	int totalLocalEdges_count = *totalLocalEdges;
	int LocalID_opv = *total_opv;
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2, wt, inst_status;
		changeEdge cE;
		sscanf(line, "%d %d %d %d", &n1, &n2, &wt, &inst_status); //edge wt is there in input file. But we don't need it. So we will ignore it.
		
		cE.inst = inst_status;

		if (Global2LocalMap.find(n1) != Global2LocalMap.end()) //if n1 is in this partition
		{
			ColWt c1, c2;
			int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID
			c1.col = first_endpoint;
			c1.flag = 0;
			cE.node1 = first_endpoint;
			if (Global2LocalMap.find(n2) != Global2LocalMap.end()) //When both n1 and n1 are in same partition
			{
				int second_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID
				cE.node2 = second_endpoint;
				if (inst_status == 1)
				{
					c2.col = second_endpoint;
					c2.flag = 0;
					
					AdjList.at(first_endpoint).push_back(c2);
					AdjList.at(second_endpoint).push_back(c1);
					totalLocalEdges_count = totalLocalEdges_count + 2;
					//if (vertexcolor[first_endpoint] == vertexcolor[second_endpoint]) //optimization //not using it, so add this logic in gpu function
					//{
						allChange_Ins.push_back(cE);
					/*}*/
					
				}
				else if (inst_status == 0) {
					allChange_Del.push_back(cE);
				}
			}
			else { //When n1 is in this partition but n2 in different partition
				if (Global2LocalMap_opv.find(n2) == Global2LocalMap_opv.end())
				{
					Global2LocalMap_opv.insert(make_pair(n2, LocalID_opv)); //opv_total is assigning LocalID for opv(other part vertex)
					Local2GlobalMap_opv.insert(make_pair(LocalID_opv, n2));
					LocalID_opv++;
				}
				cE.node2 = Global2LocalMap_opv.find(n2)->second; //*always keep node2 as the other part neighbor vertex in Ins Edge array
				c2.col = Global2LocalMap_opv.find(n2)->second;
				c2.flag = 0;
				if (inst_status == 1)
				{
					AdjList_border.at(first_endpoint).push_back(c2); //This adj list contains GlobalID of other part ngbr
					//other_part_ngbr.insert(n2); //add to other part neighbor set
					totalCrossEdges_count++;
					borderVertexFlag[first_endpoint] = 1;
					allChange_Ins_cross.push_back(cE);
					int part_n2 = PartitionID_all.at(n2);
					bvPerPart[part_n2].insert(first_endpoint);
				}
				else if (inst_status == 0) {
					allChange_Del_cross.push_back(cE);
				}
				
			}
		}
		else if (Global2LocalMap.find(n2) != Global2LocalMap.end()) //n2 in this partition but n1 in different partition
		{
			if (Global2LocalMap_opv.find(n1) == Global2LocalMap_opv.end())
			{
				Global2LocalMap_opv.insert(make_pair(n1, LocalID_opv)); //opv_total is assigning LocalID for opv(other part vertex)
				Local2GlobalMap_opv.insert(make_pair(LocalID_opv, n1));
				LocalID_opv++;
			}
			ColWt /*c1,*/ c2;
			int first_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID
			int second_endpoint = Global2LocalMap_opv.find(n1)->second; //Getting local ID of opv
			/*c1.col = first_endpoint;
			c1.flag = 0;*/
			c2.col = second_endpoint;
			c2.flag = 0;
			cE.node1 = first_endpoint;
			cE.node2 = second_endpoint; //*always keep node2 as the other part neighbor vertex in Ins Edge array
			if (inst_status == 1)
			{
				AdjList_border.at(first_endpoint).push_back(c2); //This adj list contains GlobalID of other part ngbr
				//other_part_ngbr.insert(n2); //add to other part neighbor set
				totalCrossEdges_count++;
				borderVertexFlag[first_endpoint] = 1;
				allChange_Ins_cross.push_back(cE);
				int part_n1 = PartitionID_all.at(n1);
				bvPerPart[part_n1].insert(first_endpoint);
			}
			else if (inst_status == 0) {
				allChange_Del_cross.push_back(cE);
			}
		}
	}
	*totalCrossEdges = totalCrossEdges_count; //every edge counted once
	*totalLocalEdges = totalLocalEdges_count; //every edge counted twice a->b and b->a
    *total_opv = LocalID_opv;
	fclose(graph_file);
	auto readCEstopTime = high_resolution_clock::now();//Time calculation ends
	auto readCEduration = duration_cast<microseconds>(readCEstopTime - readCEstartTime);// duration calculation
	cout << "Reading input changed edges data completed." << endl;
	cout << "Time taken to read input changed edges: " << readCEduration.count() << " microseconds" << endl;
	return;
}


/*
read_Input_Color reads input color lebel file.
accepted data format: node color
*/
void read_Input_Color(int* vertexcolor, char* myfile, map<int, int>& Global2LocalMap, /*set<int>& other_part_ngbr, map<int, int>& opVertexColorMap,*/ int* vertexcolor_opv, map<int, int>& Global2LocalMap_opv)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int node, color;
		sscanf(line, "%d %d", &node, &color);
		if (Global2LocalMap.find(node) != Global2LocalMap.end())
		{
			vertexcolor[Global2LocalMap.find(node)->second] = color;
		}
		else if (Global2LocalMap_opv.find(node) != Global2LocalMap_opv.end())
		{
			vertexcolor_opv[Global2LocalMap_opv.find(node)->second] = color;
		}
		
	}
	fclose(graph_file);

	return;
}

/*
read_graphEdges reads the original graph file
accepted data format: node1 node2 edge_weight
we consider only undirected graph here. for edge e(a,b) with weight W represented as : a b W
*/
void read_graphEdges(vector<ColList>& AdjList, vector<ColList>& AdjList_border, char* myfile, map<int, int>& Global2LocalMap, /*set<int>& other_part_ngbr,*/ int* totalLocalEdges, int* totalCrossEdges, int*& borderVertexFlag, map<int, int>& Global2LocalMap_opv, map<int, int>& Local2GlobalMap_opv, int* total_opv)
{
	cout << "Reading input graph..." << endl;
	auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	int totalCrossEdges_count = 0;
	int totalLocalEdges_count = 0;
	int opv_total = 0;
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2, wt;
		sscanf(line, "%d %d %d", &n1, &n2, &wt); //our input graph has wt. But we don't need it. So we will ignore wt.
		if (Global2LocalMap.find(n1) != Global2LocalMap.end())
		{
			ColWt /*c1,*/ c2;
			int first_endpoint = Global2LocalMap.find(n1)->second; //Getting local ID
			/*c1.col = first_endpoint;
			c1.flag = 0;*/
			if (Global2LocalMap.find(n2) != Global2LocalMap.end()) //When both n1 and n1 are in same partition
			{
				c2.col = Global2LocalMap.find(n2)->second; //Getting local ID
				c2.flag = 0;
				AdjList.at(first_endpoint).push_back(c2);
				totalLocalEdges_count++;
			}
			else {
				if (Global2LocalMap_opv.find(n2) == Global2LocalMap_opv.end())
				{
					Global2LocalMap_opv.insert(make_pair(n2, opv_total));
					Local2GlobalMap_opv.insert(make_pair(opv_total, n2));
					opv_total++;
				}
				c2.col = Global2LocalMap_opv.find(n2)->second;
				c2.flag = 0;
				AdjList_border.at(first_endpoint).push_back(c2);
				//other_part_ngbr.insert(n2); //not required
				totalCrossEdges_count++;
				borderVertexFlag[first_endpoint] = 1;
			}
		}

		
		if (Global2LocalMap.find(n2) != Global2LocalMap.end())
		{
			ColWt /*c1,*/ c2;
			int first_endpoint = Global2LocalMap.find(n2)->second; //Getting local ID
			/*c1.col = first_endpoint;
			c1.flag = 0;*/
			if (Global2LocalMap.find(n1) != Global2LocalMap.end()) //When both n1 and n1 are in same partition
			{
				c2.col = Global2LocalMap.find(n1)->second; //Getting local ID
				c2.flag = 0;
				AdjList.at(first_endpoint).push_back(c2);
				totalLocalEdges_count++;
			}
			else {
				if (Global2LocalMap_opv.find(n1) == Global2LocalMap_opv.end())
				{
					Global2LocalMap_opv.insert(make_pair(n1, opv_total)); //opv_total is assigning LocalID for opv(other part vertex)
					Local2GlobalMap_opv.insert(make_pair(opv_total, n1));
					opv_total++;
				}
				c2.col = Global2LocalMap_opv.find(n1)->second;
				c2.flag = 0;
				AdjList_border.at(first_endpoint).push_back(c2);
				//other_part_ngbr.insert(n1);//not required
				totalCrossEdges_count++;
				borderVertexFlag[first_endpoint] = 1;
			}
		}
		
	}
	*totalCrossEdges = totalCrossEdges_count; //every edge counted once
	*totalLocalEdges = totalLocalEdges_count; //every edge counted twice a->b and b->a
	*total_opv = opv_total;
	fclose(graph_file);
	auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	cout << "Reading input graph completed" << endl;
	cout << "Time taken to read input graph: " << readGraphduration.count() << " microseconds" << endl;
	return;
}



#endif