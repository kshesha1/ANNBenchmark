# ANNBenchmark
## Comparative Analysis of Faiss and Milvus: Performance Evaluation and Benchmarking of Indexing Methods of Approximate Nearest Neighbor Search (ANNS) in Vector Databases

### Research Question
Recent research has witnessed significant interest in the development and exploration of approximate nearest-neighbor search (ANNS) methods. The objective of this research is to benchmark and evaluate ANNS algorithms of two popular systems namely, Faiss (Facebook AI Similarity Search), a library for efficient similarity search and Milvus, a vector database built to power embedding similarity search. In this research we consider three ANNS algorithms across the different categories, namely, graph-based algorithms (HNSW), inverted index algorithm (IVF), and hybrid index (IVF_PQ). We also consider three evaluation methods: QPS(queries per second), Latency and number of queries comparison and build times. 

### Investigating Data Insertion and Index Creation Order
In addition to benchmarking and evaluating ANNS algorithms in Faiss and Milvus, this research investigates the order of operations involved in data insertion and index creation in Milvus standalone. The two approaches are "Insert-Then-Create-Index" and "Create-Index-Then-Insert". In the "Insert-Then-Create-Index" scenario, data is first inserted into the Milvus collection, followed by the creation of the index on the inserted data. Conversely, in the "Create-Index-Then-Insert" approach, the index is established prior to inserting the data into the collection. 

The purpose of this analysis is to identify any potential performance variances and implications associated to the order of these operations. We gain important insights into how the sequence of data insertion and index creation affects the efficiency and effectiveness of Milvus standalone by systematically comparing both strategies under the chosen ANNS algorithms (HNSW, IVF, IVF_PQ) and evaluation metrics (QPS, Latency, Per query time, Build Times).

### ANNS Algorithms Explored
#### IVF (Inverted File Index)

IVF technique used in ANN search and is based on the idea of dividing the dataset into clusters and creating an inverted index structure to store information about the vectors in each cluster.

#### HNSW (Hierarchical Navigable Small World Graphs)

HNSW builds a navigable graph where each node represents a data point in the dataset. The graph is constructed in a way that maximizes the connectivity between nearby points while maintaining a sparse structure. This allows for efficient graph traversal during the search process.

#### IVFPQ

Combines the Inverted File (IVF) and Product Quantization (PQ) techniques. First, the dataset is divided into a set of Voronoi cells using the Inverted File technique. This helps to reduce the search space. The Product Quantization technique divides the vectors into subvectors and quantizes them independently. This process reduces the memory footprint and speeds up the search process. 


### Conclusion and Analysis

We observe that the latency of Faiss was significantly lower compared to the Milvus and we reason that because Milvus creates a schema and stores the database before indexing while Fais adds the data directly to the index. In terms of QPS, we see a similar trend among the three ANN algorithms, where Milvus outperformed Faiss, indicating its ability to handle a higher volume of queries per second with increase in the number of queries. We also see a slightly bigger difference in buildtimes for IVF and HNSW, but the build times for IVFPQ are comparable. 

Our results showed that the order of insertion and index creation has a significant impact on the performance of Milvus standalone. For all three ANNS algorithms, the Create-Index-Then-Insert approach had significantly higher latency, per query time, and QPS than the Insert-Then-Create-Index approach. However, the build times for the two approaches were not as consistent. For HNSW and IVFPQ, the build times for the two approaches were similar when the number of vectors was large (1000 vectors).

The results are as shown below:

### Results
#### Faiss vs Milvus
![IVF: Latency](https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVF_Latency.png)
![IVF: QPS](https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVF%20QPS.png)
![IVF: Build times](https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVF%20Build%20times.png)

