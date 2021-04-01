#include <mpi.h>
//#include <omp.h>
#include <stdio.h>
//#include "all_structure_undir.cuh"
//#include "gpuFunctions_undir.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include<vector>
#include <chrono>
#include <map>
#include <set> 
#include <string>
#include "Supporting.h"
#include "all_structure_undir.cuh"
#include "gpuFunctions_undir.cuh"
#include "compactor.cuh"
using namespace std;

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }


/*
* Arg 1: graph file
* Arg 2: color file
* Arg 3: ChangeEdges file
* module load openmpi/4.0.3/gnu/9.2.0
* nvcc -I/share/apps/common/openmpi/4.0.3/gnu/9.2.0/include -L/share/apps/common/openmpi/4.0.3/gnu/9.2.0/lib -lmpi kernel.cu -o op
* mpirun -np 4 ./op graphFile colorFile ChangeEdgesFile > output.txt
*/


int main(int argc, char** argv) {

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    //// Initialize the MPI environment
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    //// Get the rank of this process
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    //// Get the number of processes
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
            &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }


        ////Setting a GPU for a process rank
        CUDA_RT_CALL(cudaSetDevice(rank));
        CUDA_RT_CALL(cudaFree(0));

        ////Below variables are local to each process
        int totalLocalVertices, totalLocalEdges, deviceId, numberOfSMs; //totalLocalEdges : every edges counted twice a->b b->a
        int totalCrossEdges = 0; //Total edges for which only one endpoint is in this partition. every edge counted once
        char* inputColorfile = argv[2];
        bool zeroDelFlag = false, zeroInsFlag = false, zeroDelFlag_cross = false, zeroInsFlag_cross = false;
        vector<ColList> AdjList; //stores input graph in 2D adjacency list
        vector<ColWt> AdjListFull; //Row-major implementation of adjacency list (1D)
        ColWt* AdjListFull_device; //1D array in GPU to store Row-major implementation of adjacency list 
        int* AdjListTracker_device; //1D array to track offset for each node's adjacency list
        vector<changeEdge> allChange_Ins, allChange_Del, allChange_Ins_cross, allChange_Del_cross;
        changeEdge* allChange_Ins_device; //stores all change edges marked for insertion in GPU
        changeEdge* allChange_Del_device; //stores all change edges marked for deletion in GPU
        int* counter;
        int* affected_marked;
        int* affectedNodeList;
        int* vertexcolor;
        int* previosVertexcolor;
        int* borderVertexFlag; //Stores flag(1) to indicate if a vertex is a border vertex
        vector<ColList> AdjList_border; //stores the adjlist elements which are not in same partition but neighbor of border vertices of this partition
        set<int> other_part_ngbr; //stores the neighbors from other partitions
        vector<int> PartitionID_all; //stores partition ID for all vertices

        ////Get gpu device id and number of SMs
        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
        size_t  numberOfBlocks = 32 * numberOfSMs;

        ///Read partition files in parallel. Map GlobalID -> LocalID. Map LocalID -> GlobalID 
        map<int, int> Global2LocalMap; //Used for Mapping GlobalID -> LocalID
        map<int, int> Local2GlobalMap; //Used for Mapping LocalID -> GlobalID
        string parfile = "partition" + to_string(rank) + ".txt"; //partition P reads the partition file partitionP.txt//This name is used as the output file for PuLP
        char* parfileName = new char[parfile.length() + 1];
        strcpy(parfileName, parfile.c_str());
        read_Partition_Vertices(Global2LocalMap, Local2GlobalMap, parfileName, &totalLocalVertices); //Read partition and Map

        // Print off a hello world message
        /*printf("Hello world from processor %s, rank %d out of %d processors %d GPUs. Total local vertices: %d\n",
            processor_name, rank, size, num_devices, totalLocalVertices);*/

        //Test: print map
        /*map<int, int>::iterator itr;
        for (itr = Local2GlobalMap.begin(); itr != Local2GlobalMap.end(); ++itr) {
            cout <<"rank:"<<rank<< "Local:\t" << itr->first
                << "Global:\t" << itr->second << '\n';
        }*/

        /// Read part ID for all vertices and store
        string parfileAll = "partitionAll.txt"; //This name is used as the output file for PuLP
        char* parfileAllName = new char[parfileAll.length() + 1];
        strcpy(parfileAllName, parfileAll.c_str());
        read_PartitionID_AllVertices(parfileAllName, PartitionID_all);

        //Test: print part ID
        for (int i = 0; i < 5; i++)
        {
            cout << "rank: " << rank << "vertexID: " << i << "part ID: " << PartitionID_all.at(i) << endl;
        }




        ////Read Original input graph
        AdjList.resize(totalLocalVertices);
        AdjList_border.resize(totalLocalVertices);
        borderVertexFlag = (int*)calloc(totalLocalVertices, sizeof(int));
        int* AdjListTracker = (int*)malloc((totalLocalVertices + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
        read_graphEdges(AdjList, AdjList_border, argv[1], Global2LocalMap, other_part_ngbr, &totalLocalEdges, &totalCrossEdges, borderVertexFlag);

        //Test: print other_part_ngbr
        /*set<int>::iterator itr1;
        cout << "\nThe set other_part_ngbr is : \n";
        for (itr1 = other_part_ngbr.begin(); itr1 != other_part_ngbr.end(); itr1++)
        {
            cout << "rank:" << rank<< "element:" << *itr1 << endl;
        }*/

        //Test: print total vertices and edges (both internal and cross)
        cout << "rank:" << rank << "Total Local Vertices:" << totalLocalVertices << "Total vertices from other part.:" << other_part_ngbr.size()<<endl;
        cout << "rank:" << rank << "Total Local Edges*2:" << totalLocalEdges << "Total Cross Edges:" << totalCrossEdges << endl;
        /*int i = 0;
        for ( int j =0; j < totalLocalVertices; j++)
        {
            if (i > 9)
            {
                break;
            }
            if (borderVertexFlag[j] == 1)
            {
                cout << "rank:" << rank <<" border vertex:"<< j <<endl;
                i++;
            }  
        }*/

        //Read change edges input
        readin_changes(argv[3], allChange_Ins, allChange_Del, allChange_Ins_cross, allChange_Del_cross, AdjList, /*vertexcolor,*/ Global2LocalMap, other_part_ngbr, borderVertexFlag, &totalLocalEdges, &totalCrossEdges, AdjList_border);
        int totalChangeEdges_Ins = allChange_Ins.size();
        if (totalChangeEdges_Ins == 0) {
            zeroInsFlag = true;
        }
        int totalChangeEdges_Del = allChange_Del.size();
        if (totalChangeEdges_Del == 0) {
            zeroDelFlag = true;
        }
        int totalChangeEdges_Ins_cross = allChange_Ins_cross.size();
        if (totalChangeEdges_Ins_cross == 0) {
            zeroInsFlag_cross = true;
        }
        int totalChangeEdges_Del_cross = allChange_Del_cross.size();
        if (totalChangeEdges_Del_cross == 0) {
            zeroDelFlag_cross = true;
        }

        //Test: print size of Ins and Del edges
        cout << "After reading changed edges:" << endl;
        cout << "rank:" << rank << "Total Ins Edges(local):" << totalChangeEdges_Ins << " Total Del Edges(local):" << totalChangeEdges_Del << endl;
        cout << "rank:" << rank << "Total Ins Edges(cross):" << totalChangeEdges_Ins_cross << " Total Del Edges(cross):" << totalChangeEdges_Del_cross << endl;
        cout << "rank:" << rank << "Total Local Edges*2(after CE):" << totalLocalEdges << "Total Cross Edges(after CE):" << totalCrossEdges << endl;

        ////read input vertex color label
        map<int, int> opVertexColorMap; //Stores color for the Other Part neighbor Vertex(opVertex)
        CUDA_RT_CALL(cudaMallocManaged(&vertexcolor, totalLocalVertices * sizeof(int)));
        read_Input_Color(vertexcolor, inputColorfile, Global2LocalMap, other_part_ngbr, opVertexColorMap);

        //Test: print initial 10 colors and colors of opVertices
        /*for (int i = 0; i< 10; i++) {
            cout << "rank:" << rank << "Color:\t" << vertexcolor[i]<<endl;
        }
        map<int, int>::iterator itr2;
        for (itr2 = opVertexColorMap.begin(); itr2 != opVertexColorMap.end(); ++itr2) {
            cout << "rank:" << rank << "opVertex:\t" << itr2->first
                << "Color:\t" << itr2->second << '\n';
        }*/


        ////Transfer input graph, changed edges to GPU and set memory advices
        transfer_data_to_GPU(AdjList, AdjListTracker, AdjListFull, AdjListFull_device,
            totalLocalVertices, totalLocalEdges, AdjListTracker_device, zeroInsFlag,
            allChange_Ins, allChange_Ins_device, totalChangeEdges_Ins,
            deviceId, totalChangeEdges_Del, zeroDelFlag, allChange_Del_device,
            counter, affected_marked, affectedNodeList, previosVertexcolor,/*updatedAffectedNodeList_del, updated_counter_del,*/ allChange_Del, numberOfBlocks);

        

        ////Initialize supporting variables
        int* change = 0;
        CUDA_RT_CALL(cudaMallocManaged(&change, sizeof(int)));
        
        ////process change edges////
        ////Process del edges
        if (zeroDelFlag != true) {
            auto startTimeDelEdge = high_resolution_clock::now(); //Time calculation start

            deleteEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, vertexcolor, previosVertexcolor, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device, affected_marked, change);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

            //auto stopTimeDelEdge = high_resolution_clock::now();//Time calculation ends
            //auto durationDelEdge = duration_cast<microseconds>(stopTimeDelEdge - startTimeDelEdge);// duration calculation
            //cout << "**Time taken for processing del edges: "
            //    << float(durationDelEdge.count()) / 1000 << " milliseconds**" << endl;
            //total_time += float(durationDelEdge.count()) / 1000;
        }

        ////Process ins edges
        if (zeroInsFlag != true) {
            auto startTimeInsEdge = high_resolution_clock::now(); //Time calculation start

            insEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Ins_device, vertexcolor, previosVertexcolor, totalChangeEdges_Ins, AdjListFull_device, AdjListTracker_device, affected_marked, change);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

            //auto stopTimeInsEdge = high_resolution_clock::now();//Time calculation ends
            //auto durationInsEdge = duration_cast<microseconds>(stopTimeInsEdge - startTimeInsEdge);// duration calculation
            //cout << "**Time taken for processing ins edges: "
            //    << float(durationInsEdge.count()) / 1000 << " milliseconds**" << endl;
            //total_time += float(durationInsEdge.count()) / 1000;
        }

        //auto startTimeDelNeig = high_resolution_clock::now(); //Time calculation start

        //we use compactor in place of just adding directly using atomic fn to avoid duplication of affected vertices in list
        *counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, totalLocalVertices, predicate(), THREADS_PER_BLOCK);
        //recolor affected neighbors
        while (*change > 0)
        {
            *change = 0;
            //reset affected_del to 0
            CUDA_RT_CALL(cudaMemset(affected_marked, 0, totalLocalVertices * sizeof(int)));
            //printf("after memset 0: affected_del flag for %d = %d \n", 1, affected_del[1]);

            //find eligible neighbors which should be updated
            findEligibleNeighbors << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, AdjListFull_device, AdjListTracker_device, affected_marked, previosVertexcolor, vertexcolor, counter);
            CUDA_RT_CALL(cudaGetLastError());
            //find the next frontier: it collects the vertices to be recolored and store without duplicate in affectedNodeList
            *counter = cuCompactor::compact<int, int>(affected_marked, affectedNodeList, totalLocalVertices, predicate(), THREADS_PER_BLOCK);
            /*printf("After findEligibleNeighbors: affectedNodeList_del elements:\n");
            for (int i = 0; i < *counter_del; i++)
            {
                printf("%d:", affectedNodeList_del[i]);
            }*/
            CUDA_RT_CALL(cudaMemset(affected_marked, 0, totalLocalVertices * sizeof(int))); //new
            //recolor the eligible neighbors
            recolorNeighbor << < numberOfBlocks, THREADS_PER_BLOCK >> > (affectedNodeList, vertexcolor, previosVertexcolor, AdjListFull_device, AdjListTracker_device, affected_marked, counter, change);
            CUDA_RT_CALL(cudaGetLastError()); 
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }
        //auto stopTimeDelNeig = high_resolution_clock::now();//Time calculation ends
        //auto durationDelNeig = duration_cast<microseconds>(stopTimeDelNeig - startTimeDelNeig);// duration calculation
        //cout << "**Time taken for processing affected neighbors: "
        //    << float(durationDelNeig.count()) / 1000 << " milliseconds**" << endl;
        //total_time += float(durationDelNeig.count()) / 1000;
        //cout << "****Total Time for Vertex Color Update: "
        //    << total_time << " milliseconds****" << endl;





        ////deleteallocated memory
        //delete[] parfileName;
        if (zeroDelFlag != true) {
            CUDA_RT_CALL(cudaFree(allChange_Del_device));
        }
        if (zeroInsFlag != true) {
            CUDA_RT_CALL(cudaFree(allChange_Ins_device));
        }
        //CUDA_RT_CALL(cudaFree(change));
        CUDA_RT_CALL(cudaFree(vertexcolor));
        CUDA_RT_CALL(cudaFree(affected_marked));
        CUDA_RT_CALL(cudaFree(affectedNodeList));
        CUDA_RT_CALL(cudaFree(counter));
        CUDA_RT_CALL(cudaFree(AdjListFull_device));
        CUDA_RT_CALL(cudaFree(AdjListTracker_device));
        CUDA_RT_CALL(cudaFree(previosVertexcolor));
    // Finalize the MPI environment.
    MPI_Finalize();
}