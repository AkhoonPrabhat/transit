#include <omp.h>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph{
    
        int V;
        vector<vector<int>> adj;

        public:
            Graph(int V): V(V), adj(V){}
            
            void addEdge(int u, int v){
                adj[u].push_back(v);
            }

            void parallelDFS(int startVertex){
                vector<bool> visited(V, false);
                parallelDFSUtil(startVertex, visited);  
            }

            void parallelDFSUtil(int vertex, vector<bool>& visited){

                visited[vertex] = true;

                cout << vertex << " ";

                #pragma omp parallel for
                for (int v = 0; v < adj[vertex].size(); v++){
                    if (!visited[adj[vertex][v]]){
                        parallelDFSUtil(adj[vertex][v], visited);
                    }
                }   
            }

            void parallelBFS(int startVertex){
                vector<bool> visited(V, false);
                visited[startVertex] = true;

                queue<int> q;
                q.push(startVertex);

                
                while (!q.empty()){
                    int vertex = q.front();
                    q.pop();
                    cout << vertex << " ";

                    #pragma omp parallel for
                    for(int v = 0; v < adj[vertex].size(); v++){
                        if(!visited[adj[vertex][v]]){
                            q.push(adj[vertex][v]);
                        }
                    }
                }
                


            }


    
            

};



int main(){

    Graph g(7);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    
    /*
        0 -------->1
        |         / \
        |        /   \
        |       /     \
        v       v       v
        2 ----> 3       4
        |      |
        |      |
        v      v
        5      6
    */

    cout << "Depth-First Search (DFS): ";
    g.parallelDFS(0);
    cout << endl;

    cout << "Breadth-First Search (BFS): ";
    g.parallelBFS(0);
    cout << endl;


    return 0;
}