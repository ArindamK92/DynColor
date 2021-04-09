Change our format graph to Metis format:
g++ -o op CovertToMetisFormat.cpp
./op graph.txt 4 > graph_metis