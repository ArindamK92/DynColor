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
#include <omp.h>
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
* 
* Note: opv - other part vertex is a vertex adjacent to a border vertex(which is inside local process)
* crossing edges - edges that has two endpoints in two different partitions
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
        vector<ColWt> AdjListFull_border; //Row-major implementation of adjacency list (1D)
        ColWt* AdjListFull_device; //1D array in GPU to store Row-major implementation of adjacency list 
        ColWt* AdjListFull_border_device; //1D array in GPU to store Row-major implementation of adjacency list
        int* AdjListTracker;
        int* AdjListTracker_device; //1D array to track offset for each node's adjacency list
        int* AdjListTracker_border;
        int* AdjListTracker_border_device; //1D array to track offset for each border node's opv adjacency list
        vector<changeEdge> allChange_Ins, allChange_Del, allChange_Ins_cross, allChange_Del_cross;
        changeEdge* allChange_Ins_device; //stores all change edges marked for insertion in GPU
        changeEdge* allChange_Del_device; //stores all change edges marked for deletion in GPU
        changeEdge* allChange_Ins_cross_device; //stores all change edges(crossing edges) marked for insertion in GPU
        changeEdge* allChange_Del_cross_device; //stores all change edges(crossing edges) marked for deletion in GPU
        int* counter;
        int* counter_border;
        int* affected_marked;
        int* affected_marked_border;
        int* affectedNodeList;
        int* affectedNodeList_border;
        int* vertexcolor; //Stores vertex color for all vertices inside this partition
        int* vertexcolor_opv; //Stores vertex color for opv
        int* previosVertexcolor;
        int* previosVertexcolor_opv;
        int* borderVertexFlag; //Stores flag(1) to indicate if a vertex is a border vertex
        vector<ColList> AdjList_border; //stores the adjlist elements which are not in same partition but neighbor of border vertices of this partition
        //set<int> other_part_ngbr; //stores the neighbors from other partitions
        vector<int> PartitionID_all; //stores partition ID for all vertices
        map<int, int> Global2LocalMap; //Used for Mapping GlobalID -> LocalID for local vertices
        map<int, int> Local2GlobalMap; //Used for Mapping LocalID -> GlobalID for local vertices
        map<int, int> Global2LocalMap_opv; //Used for Mapping GlobalID -> LocalID for other part vertices
        map<int, int> Local2GlobalMap_opv; //Used for Mapping LocalID -> GlobalID for other part vertices
        int total_opv = 0; //Total other part vertex adjacent to some vertex of this partition
        set<int>* bvPerPart = new set<int>[size]; //stores border vertices adjacent to each partition
        int* SC_size_border; //Stores saturation color set size for border vertices
        int* SC_size_opv; //Stores saturation color set size for other part vertices
        int** buff = new int*[size]; //2D array. ith row stores the elements need to send to part i
        int** buff_recv = new int* [size]; //2D array. ith row stores the elements received from part i
        int* recv_item_count = new int [size]; //stores how many items received from ith part

        ////Get gpu device id and number of SMs
        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
        size_t  numberOfBlocks = 32 * numberOfSMs;

        ///Read partition files in parallel. Map GlobalID -> LocalID. Map LocalID -> GlobalID 
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
        read_PartitionID_AllVertices(parfileAllName, PartitionID_all); //stores partIDs for GlobalIDs of vertices

        //Test: print part ID
       /* for (int i = 0; i < 5; i++)
        {
            cout << "rank: " << rank << "vertexID: " << i << "part ID: " << PartitionID_all.at(i) << endl;
        }*/


        ////Read Original input graph
        AdjList.resize(totalLocalVertices);
        AdjList_border.resize(totalLocalVertices);
        /*borderVertexFlag = (int*)calloc(totalLocalVertices, sizeof(int));*/
        CUDA_RT_CALL(cudaMallocManaged(&borderVertexFlag, totalLocalVertices * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(borderVertexFlag, 0, totalLocalVertices * sizeof(int)));
        AdjListTracker = (int*)malloc((totalLocalVertices + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
        AdjListTracker_border = (int*)malloc((totalLocalVertices + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
        read_graphEdges(AdjList, AdjList_border, argv[1], Global2LocalMap, /*other_part_ngbr,*/ &totalLocalEdges, &totalCrossEdges, borderVertexFlag, Global2LocalMap_opv, Local2GlobalMap_opv, &total_opv);

        //Test: print other_part_ngbr
        /*set<int>::iterator itr1;
        cout << "\nThe set other_part_ngbr is : \n";
        for (itr1 = other_part_ngbr.begin(); itr1 != other_part_ngbr.end(); itr1++)
        {
            cout << "rank:" << rank<< "element:" << *itr1 << endl;
        }*/

        //Test: print total vertices and edges (both internal and cross)
        /*cout << "rank:" << rank << "Total Local Vertices:" << totalLocalVertices << "Total vertices from other part.:" << other_part_ngbr.size()<<endl;
        cout << "rank:" << rank << "Total Local Edges*2:" << totalLocalEdges << "Total Cross Edges:" << totalCrossEdges << endl;*/
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
        readin_changes(argv[3], allChange_Ins, allChange_Del, allChange_Ins_cross, allChange_Del_cross, AdjList, /*vertexcolor,*/ Global2LocalMap, /*other_part_ngbr,*/ borderVertexFlag, &totalLocalEdges, &totalCrossEdges, AdjList_border, Global2LocalMap_opv, Local2GlobalMap_opv, &total_opv, PartitionID_all, bvPerPart);
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

        
        //Test: print bvPerPart
        //set<int>::iterator itr;
        //for (itr = bvPerPart[0].begin(); itr != bvPerPart[0].end(); itr++) //printing elements only for part 0
        //{
        //    cout << "rank: " << rank << "element: "<< *itr << endl;
        //}

        //Test: print size of Ins and Del edges
        /*cout << "After reading changed edges:" << endl;
        cout << "rank:" << rank << "Total OPVertex:" << total_opv << endl;
        cout << "rank:" << rank << "Total Ins Edges(local):" << totalChangeEdges_Ins << " Total Del Edges(local):" << totalChangeEdges_Del << endl;
        cout << "rank:" << rank << "Total Ins Edges(cross):" << totalChangeEdges_Ins_cross << " Total Del Edges(cross):" << totalChangeEdges_Del_cross << endl;
        cout << "rank:" << rank << "Total Local Edges*2(after CE):" << totalLocalEdges << "Total Cross Edges(after CE):" << totalCrossEdges << endl;*/

        //Compute total border vertices and store them in borderVertexList
        int* borderVertexList;
        CUDA_RT_CALL(cudaMallocManaged(&borderVertexList, totalLocalVertices * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(borderVertexList, 0, totalLocalVertices * sizeof(int)));
        int total_borderVertex = cuCompactor::compact<int, int>(borderVertexFlag, borderVertexList, totalLocalVertices, predicate(), THREADS_PER_BLOCK);

        //Test: print border vertices
        /*cout << "rank: " << rank << "total border vertices: " << total_borderVertex << endl;
        for (int i = 0; i < 5; i++) {
            cout << "rank: " << rank << "border vertex ID: " << borderVertexList[i] << endl;
        }*/

        ////read input vertex color label
        //map<int, int> opVertexColorMap; //Stores color for the Other Part neighbor Vertex(opVertex)
        CUDA_RT_CALL(cudaMallocManaged(&vertexcolor, totalLocalVertices * sizeof(int)));
        CUDA_RT_CALL(cudaMallocManaged(&vertexcolor_opv, total_opv * sizeof(int)));
        read_Input_Color(vertexcolor, inputColorfile, Global2LocalMap, /*other_part_ngbr, opVertexColorMap,*/ vertexcolor_opv, Global2LocalMap_opv);

        //Test: print initial 10 colors and colors of opVertices
        /*for (int i = 0; i< 5; i++) {
            cout << "rank: " << rank << "opvertexGlobal ID: " << Local2GlobalMap_opv.find(i)->second << "original partID: " << PartitionID_all.at(Local2GlobalMap_opv.find(i)->second) <<"opvColor:\t" << vertexcolor_opv[i]<<endl;
        }*/
        


        ////Transfer input graph, changed edges to GPU and set memory advices
        transfer_data_to_GPU(AdjList, AdjListTracker, AdjListFull, AdjListFull_device,
            totalLocalVertices, totalLocalEdges, AdjListTracker_device, zeroInsFlag,
            allChange_Ins, allChange_Ins_device, totalChangeEdges_Ins,
            deviceId, totalChangeEdges_Del, zeroDelFlag, allChange_Del_device,
            counter, affected_marked, affectedNodeList, previosVertexcolor,/*updatedAffectedNodeList_del, updated_counter_del,*/ allChange_Del, numberOfBlocks);

        transfer_border_data_to_GPU(AdjList_border, AdjListTracker_border, AdjListFull_border, AdjListFull_border_device,
            totalLocalVertices, totalCrossEdges, AdjListTracker_border_device, zeroInsFlag_cross,
            allChange_Ins_cross, allChange_Ins_cross_device, totalChangeEdges_Ins_cross,
            deviceId, totalChangeEdges_Del_cross, zeroDelFlag_cross, allChange_Del_cross_device,
            counter_border, affected_marked_border, affectedNodeList_border, previosVertexcolor_opv,/*updatedAffectedNodeList_del, updated_counter_del,*/ allChange_Del_cross, numberOfBlocks, total_opv); //Added total_opv

        ////Compute SC size for all border verties and store in SC_size_border
        CUDA_RT_CALL(cudaMallocManaged(&SC_size_border, totalLocalVertices * sizeof(int)));
        CUDA_RT_CALL(cudaMemset(SC_size_border, 0, totalLocalVertices * sizeof(int)));
        CUDA_RT_CALL(cudaMallocManaged(&SC_size_opv, total_opv * sizeof(int))); 
        CUDA_RT_CALL(cudaMemset(SC_size_opv, 0, total_opv * sizeof(int))); //initializing SC_size_opv
        compute_SC_size << < numberOfBlocks, THREADS_PER_BLOCK >> > (total_borderVertex, borderVertexList, SC_size_border, AdjListFull_device, AdjListTracker_device, AdjListFull_border_device, AdjListTracker_border_device, vertexcolor, vertexcolor_opv);
        
        //Test: print size of SC for border vertices
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required
        //for (int i = 0; i < 5; i++) {
        //    cout << "rank: " << rank << "size of SC: " << SC_size_border[borderVertexList[i]] << endl; //will print 0 is it is not a border vertex
        //}
        

        ////Initialize supporting variables
        int* change = 0;
        CUDA_RT_CALL(cudaMallocManaged(&change, sizeof(int)));

        float total_time = 0;
        //auto startTime = high_resolution_clock::now(); //Time calculation start
        ////process change edges////
        ////Process del edges(internal)
        if (zeroDelFlag != true) {
            auto startTimeDelEdge = high_resolution_clock::now(); //Time calculation start

            deleteEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, vertexcolor, previosVertexcolor, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device, affected_marked, change, borderVertexFlag, vertexcolor_opv, AdjListFull_border_device, AdjListTracker_border_device, affected_marked_border);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

            auto stopTimeDelEdge = high_resolution_clock::now();//Time calculation ends
            auto durationDelEdge = duration_cast<microseconds>(stopTimeDelEdge - startTimeDelEdge);// duration calculation
            cout << "rank: " << rank << "**Time taken for processing del edges(internal): "
                << float(durationDelEdge.count()) / 1000 << " milliseconds**" << endl;
            total_time += float(durationDelEdge.count()) / 1000;
        }
        ////Process del edges(cross)
        if (zeroDelFlag_cross != true) {
            auto startTimeDelEdge = high_resolution_clock::now(); //Time calculation start

            deleteEdge_cross << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_cross_device, vertexcolor, previosVertexcolor, vertexcolor_opv, /*previosVertexcolor_opv,*/ totalChangeEdges_Del_cross, AdjListFull_device, AdjListTracker_device, AdjListFull_border_device, AdjListTracker_border_device, affected_marked, affected_marked_border, change);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

            /*cout << "rank: " << rank << "deleteEdge_cross done" << endl;*/
            auto stopTimeDelEdge = high_resolution_clock::now();//Time calculation ends
            auto durationDelEdge = duration_cast<microseconds>(stopTimeDelEdge - startTimeDelEdge);// duration calculation
            cout << "rank: " << rank << "**Time taken for processing del edges(cross): "
                << float(durationDelEdge.count()) / 1000 << " milliseconds**" << endl;
            total_time += float(durationDelEdge.count()) / 1000;
        }

        ////Process ins edges(internal)
        if (zeroInsFlag != true) {
            auto startTimeInsEdge = high_resolution_clock::now(); //Time calculation start

            insEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Ins_device, vertexcolor, previosVertexcolor, totalChangeEdges_Ins, AdjListFull_device, AdjListTracker_device, affected_marked, change, borderVertexFlag, vertexcolor_opv, AdjListFull_border_device, AdjListTracker_border_device, affected_marked_border);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

            auto stopTimeInsEdge = high_resolution_clock::now();//Time calculation ends
            auto durationInsEdge = duration_cast<microseconds>(stopTimeInsEdge - startTimeInsEdge);// duration calculation
            cout << "rank: " << rank << "**Time taken for processing ins edges(internal): "
                << float(durationInsEdge.count()) / 1000 << " milliseconds**" << endl;
            total_time += float(durationInsEdge.count()) / 1000;
        }


        ////Prepare send buffers with GlobalID of border vertices and related SC_size
        //**Note: We are sending SC_size for only those border vertices which are the endpoints of some Ins edges. We are NOT sending SC_size for all border vertices
        auto startTimePrepBuf = high_resolution_clock::now(); //Time calculation start
        for (int j = 0; j < size; j++)
        {
            if (j != rank)
            {
                int buff_size = bvPerPart[j].size();
                int* buff_arr = new int[2 * buff_size]; //2*buff_size is required as it stores all Global vertex ID and corresponding SC_size
                std::vector<int> vec(buff_size);
                std::copy(bvPerPart[j].begin(), bvPerPart[j].end(), vec.begin());

#pragma omp parallel num_threads(32) //Taking fixed 8 threads
                {
#pragma omp for schedule(dynamic,1) //Dynamic scheduling in openMP
                    for (int i = 0; i < buff_size; i++) {
                        int v = vec.at(i);
                        int globalID = Local2GlobalMap.find(v)->second;
                        int scSize = SC_size_border[v];
                        buff_arr[i] = globalID; //ith element is the global ID
                        buff_arr[i + buff_size] = scSize; // (i + buff_size)th element is the sc_size
                        /*buff[j] = buff_arr;*/
                    }
                }
                buff[j] = buff_arr;
            }
        }
        auto stopTimePrepBuf = high_resolution_clock::now();//Time calculation ends
        auto durationPrepBuf = duration_cast<microseconds>(stopTimePrepBuf - startTimePrepBuf);// duration calculation
        cout << "rank: " << rank << "**Time taken for preparing SC_size related buffer: "
            << float(durationPrepBuf.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationPrepBuf.count()) / 1000;


        //Test: print buff
       /* if (rank != 0)
        {
            for (int i = 0; i < 5; i++)
            {
                int buff_size = bvPerPart[0].size();
                cout << "rank:" << rank << "buff GlobalID:" << buff[0][i] << "buff value:" << buff[0][buff_size + i] << endl;
            }
        }*/
        
        ////Send SC_size for the local border vertices and receive SC_size for opv. Note: Border vertex in this part is opv in some other part
        auto startTimeSendRecv = high_resolution_clock::now(); //Time calculation start
#pragma omp parallel num_threads(size)
        {
            int thread_id = omp_get_thread_num();
            if (rank != thread_id)
            {
                int buff_size = 2*bvPerPart[thread_id].size();
                if (buff_size != 0)
                {
                    MPI_CALL(MPI_Send(buff[thread_id], buff_size, MPI_INT, thread_id, 0, MPI_COMM_WORLD));
                    MPI_Status status;
                    MPI_CALL(MPI_Probe(thread_id, 0, MPI_COMM_WORLD, &status)); //MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status)
                    int n_items;
                    MPI_CALL(MPI_Get_count(&status, MPI_INT, &n_items));
                    int* recv_buf = new int[n_items];
                    MPI_CALL(MPI_Recv(recv_buf, n_items, MPI_INT, thread_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)); //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status)
                    
                    //Test: Print received buffer : received format- ith element contains GlobalID of opv, (n_items/2 + i)th element contains related SC_size
                    /*for (int i = 0; i < 5; i++)
                    {
                        cout << "rank:" << rank << "received from part: " << thread_id << "buff GlobalID_opv:" << recv_buf[i] << "buff value:" << recv_buf[n_items/2 + i] << endl;
                    }*/

                    //new addition: commented
                    //for (int i = 0; i < n_items / 2; i++) //****might need to make separate omp parallel for module
                    //{
                    //    int localID_opv = Global2LocalMap_opv.find(recv_buf[i])->second;
                    //    int scSize_opv = recv_buf[n_items / 2 + i];
                    //    SC_size_opv[localID_opv] = scSize_opv;

                    //    //Test
                    //    /*if (localID_opv == 180651)
                    //    {
                    //        cout << "rank:" << rank << "Global ID: " << recv_buf[i] << "localID_opv: " << localID_opv << "scSize_opv: " << scSize_opv << "SC_size_opv: "<< SC_size_opv[localID_opv]<<endl;
                    //    }*/
                    //    
                    //}
                    //new addition
                    recv_item_count[thread_id] = n_items;
                    buff_recv[thread_id] = recv_buf;
                    
                }
                
            }
            
        }

        //new addition
        for (int j = 0; j < size; j++)
        {
            if (j != rank)
            {
#pragma omp parallel num_threads(32) //Taking fixed 8 threads
                {
#pragma omp for schedule(dynamic,1) //Dynamic scheduling in openMP
                    for (int i = 0; i < recv_item_count[j] / 2; i++) //****might need to make separate omp parallel for module
                    {
                        int localID_opv = Global2LocalMap_opv.find(buff_recv[j][i])->second;
                        int scSize_opv = buff_recv[j][recv_item_count[j] / 2 + i];
                        SC_size_opv[localID_opv] = scSize_opv;

                        //Test
                        /*if (localID_opv == 180651)
                        {
                            cout << "rank:" << rank << "Global ID: " << recv_buf[i] << "localID_opv: " << localID_opv << "scSize_opv: " << scSize_opv << "SC_size_opv: "<< SC_size_opv[localID_opv]<<endl;
                        }*/

                    }
                }
            }
        }

        auto stopTimeSendRecv = high_resolution_clock::now();//Time calculation ends
        auto durationSendRecv = duration_cast<microseconds>(stopTimeSendRecv - startTimeSendRecv);// duration calculation
        cout << "rank: " << rank << "**Time taken for send-recv SC_size: "
            << float(durationSendRecv.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationSendRecv.count()) / 1000;
        //Test:
        //Will show 0 mostly as we are NOT sending SC_size for all border vertices. So only for few opv SC_size_opv will be non zero while testing.
        /*for (int i = 180651; i < 180661; i++)
        {
            cout << "rank:" << rank << "SC_size_opv: " << SC_size_opv[i] <<  endl;
        }*/

        
        ////Process ins edges(cross)
        if (zeroInsFlag_cross != true) {
            auto startTimeInsEdge = high_resolution_clock::now(); //Time calculation start

            insEdge_cross << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Ins_cross_device, vertexcolor, previosVertexcolor, vertexcolor_opv, totalChangeEdges_Ins_cross, AdjListFull_device, AdjListTracker_device, AdjListFull_border_device, AdjListTracker_border_device, affected_marked, affected_marked_border, change, SC_size_border, SC_size_opv);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize()); //comment this if required

            //cout << "rank: " << rank << "insEdge_cross done" << endl;
            auto stopTimeInsEdge = high_resolution_clock::now();//Time calculation ends
            auto durationInsEdge = duration_cast<microseconds>(stopTimeInsEdge - startTimeInsEdge);// duration calculation
            cout << "rank: " << rank << "**Time taken for processing ins edges(cross): "
                << float(durationInsEdge.count()) / 1000 << " milliseconds**" << endl;
            total_time += float(durationInsEdge.count()) / 1000;
        }
        auto startTimeDelNeig = high_resolution_clock::now(); //Time calculation start

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
        auto stopTimeDelNeig = high_resolution_clock::now();//Time calculation ends
        auto durationDelNeig = duration_cast<microseconds>(stopTimeDelNeig - startTimeDelNeig);// duration calculation
        cout << "rank: " << rank << "**Time taken for processing affected neighbors: "
            << float(durationDelNeig.count()) / 1000 << " milliseconds**" << endl;
        total_time += float(durationDelNeig.count()) / 1000;
        cout << "rank: " << rank << "****Total Time for Vertex Color Update: "
            << total_time << " milliseconds****" << endl;

        //auto stopTime = high_resolution_clock::now();//Time calculation ends
        //auto duration = duration_cast<microseconds>(stopTime - startTime);// duration calculation
        //cout << "rank: " << rank <<"**Time taken Total: "
        //    << float(duration.count()) / 1000 << " milliseconds**" << endl;


        ////deleteallocated memory
        //delete[] parfileName;
        if (zeroDelFlag != true) {
            CUDA_RT_CALL(cudaFree(allChange_Del_device));
        }
        if (zeroInsFlag != true) {
            CUDA_RT_CALL(cudaFree(allChange_Ins_device));
        }
        if (zeroDelFlag_cross != true) {
            CUDA_RT_CALL(cudaFree(allChange_Del_cross_device));
        }
        if (zeroInsFlag_cross != true) {
            CUDA_RT_CALL(cudaFree(allChange_Ins_cross_device));
        }

        CUDA_RT_CALL(cudaFree(vertexcolor));
        CUDA_RT_CALL(cudaFree(vertexcolor_opv));
        CUDA_RT_CALL(cudaFree(previosVertexcolor));
        CUDA_RT_CALL(cudaFree(previosVertexcolor_opv));
        CUDA_RT_CALL(cudaFree(affected_marked));
        CUDA_RT_CALL(cudaFree(affected_marked_border));
        CUDA_RT_CALL(cudaFree(affectedNodeList));
        CUDA_RT_CALL(cudaFree(affectedNodeList_border));
        CUDA_RT_CALL(cudaFree(counter));
        CUDA_RT_CALL(cudaFree(counter_border));
        CUDA_RT_CALL(cudaFree(AdjListFull_device));
        CUDA_RT_CALL(cudaFree(AdjListFull_border_device));
        CUDA_RT_CALL(cudaFree(AdjListTracker_device));
        CUDA_RT_CALL(cudaFree(AdjListTracker_border_device));
        delete AdjListTracker;
        delete AdjListTracker_border;
        CUDA_RT_CALL(cudaFree(borderVertexFlag));
        CUDA_RT_CALL(cudaFree(borderVertexList));
        CUDA_RT_CALL(cudaFree(SC_size_border));
        delete buff;
    // Finalize the MPI environment.
    MPI_Finalize();
}