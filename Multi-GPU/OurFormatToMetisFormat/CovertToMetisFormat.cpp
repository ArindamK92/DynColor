// CovertToMetisFormat.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <vector>
#include "all_structure.h"

using namespace std;


/*
Our graph format starts from vertex 0
It accepts a weighted graph format. But does not use weight. Can be modified to accept unweighted format easily
Accepted format: a b wt
E.g.:
0 1 3
1 3 5
1 2 4
Output format starts from vertex 1. Output is in Metis format. First line is <nodes> <edges>
E.g.
4 3
2
1 4 3
2
2
*/


int main(int argc, char* argv[])
{
	int nodes = atoi(argv[2]);
	vector<ColList> AdjList; //stores input graph in 2D adjacency list
	AdjList.resize(nodes);
	int* AdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
	int edges = read_graphEdges(AdjList, argv[1]);
	int x = 0;

	cout << nodes << " " << edges << endl;
	for (int i = 0; i < nodes; i++)
	{
		int list_size = AdjList.at(i).size();
		for (int j = 0; j < list_size; j++)
		{
			if (x == 0)
			{
				cout << (AdjList.at(i).at(j).col + 1); //we add 1 as our format starts from 0
				x++;
			}
			else {
				cout << " " << (AdjList.at(i).at(j).col + 1); //we add 1 as our format starts from 0
			}
		}
		cout << endl;
		x = 0;
	}

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
