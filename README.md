# ANNBenchmark
## Comparative Analysis of Faiss and Milvus: Performance Evaluation and Benchmarking of Indexing Methods of Approximate Nearest Neighbor Search (ANNS) in Vector Databases

### Research Question
Recent research has witnessed significant interest in the development and exploration of approximate nearest-neighbor search (ANNS) methods. The objective of this research is to benchmark and evaluate ANNS algorithms of two popular systems namely, Faiss (Facebook AI Similarity Search), a library for efficient similarity search and Milvus, a vector database built to power embedding similarity search. In this research we consider three ANNS algorithms across the different categories, namely, graph-based algorithms (HNSW), inverted index algorithm (IVF), and hybrid index (IVF_PQ). We also consider three evaluation methods: QPS(queries per second), Latency and number of queries comparison and build times. 

### Investigating Data Insertion and Index Creation Order
In addition to benchmarking and evaluating ANNS algorithms in Faiss and Milvus, this research investigates the order of operations involved in data insertion and index creation in Milvus standalone. The two approaches are "Insert-Then-Create-Index" and "Create-Index-Then-Insert". In the "Insert-Then-Create-Index" scenario, data is first inserted into the Milvus collection, followed by the creation of the index on the inserted data. Conversely, in the "Create-Index-Then-Insert" approach, the index is established prior to inserting the data into the collection. 

The purpose of this analysis is to identify any potential performance variances and implications associated to the order of these operations. We gain important insights into how the sequence of data insertion and index creation affects the efficiency and effectiveness of Milvus standalone by systematically comparing both strategies under the chosen ANNS algorithms (HNSW, IVF, IVF_PQ) and evaluation metrics (QPS, Latency, Per query time, Build Times).

### Database and Hardware

The database used in this study was the SIFT1M dataset, which is a collection of 1 million 128-dimensional SIFT features. Due to memory constraints, only 10,000 samples were used.

The hardware used in this study was a Docker instance with 7.49GB of container memory and 8 cores allocated. The processor was an i7-CPU with 4 cores. The RAM was 16GB.

The software used in this study was Faiss-cpu and Milvus version 2.2.13.


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
### Faiss vs Milvus
#### IVF

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVF%20Build%20times.png"
    alt="IVF: Build times">
  <br>
  IVF: Build times
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVF_Latency.png"
    alt="IVF: Latency">
  <br>
  IVF: Latency
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVF%20QPS.png"
    alt="IVF: QPS">
  <br>
  IVF: QPS
</p>



#### HNSW
<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/HNSW%20Build%20times.png"
    alt="HNSW: Build times">
  <br>
  HNSW: Build times
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/HNSW%20Latency.png"
    alt="HNSW: Latency">
  <br>
  HNSW: Latency
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/HNSW%20QPS.png"
    alt="HNSW: QPS">
  <br>
  HNSW: QPS
</p>

#### IVF_PQ

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVFPQ%20Build%20times.png"
    alt="IVFPQ: Build times">
  <br>
  IVFPQ: Build times
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVFPQ%20Latency.png"
    alt="IVFPQ: Latency">
  <br>
  IVFPQ: Latency
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Faiss%20vs%20Milvus/IVFPQ_QPS.png"
    alt="IVFPQ: QPS">
  <br>
  IVFPQ: QPS
</p>

### Milvus: Insert-Then-Create-Index vs Create-Index-Then-Insert
#### IVF (Milvus Index Operation)
<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVF_buildtimes.png"
    alt="IVF: Build times">
  <br>
  IVF: Build times
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVF_latency_PQT.png"
    alt="IVF: Latency and Per query time">
  <br>
  IVF: Latency and Per query time
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVF_QPS.png"
    alt="IVF: QPS">
  <br>
  IVF: QPS
</p>

#### HNSW (Milvus Index Operation)
<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/hnsw_buildtimes.png"
    alt="HNSW: Build times">
  <br>
  HNSW: Build times
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/hnsw_latency.png"
    alt="HNSW: Latency">
  <br>
  HNSW: Latency
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/hnsw_PQT.png"
    alt="HNSW: Per query time">
  <br>
  HNSW: Per query time
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/hnsw_QPS.png"
    alt="HNSW: QPS">
  <br>
  HNSW: QPS
</p>

#### IVFPQ (Milvus Index Operation)
<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVFPQ_Build%20times.png"
    alt="IVFPQ: Build times">
  <br>
  IVFPQ: Build times
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVFPQ_latency.png"
    alt="IVFPQ: Latency">
  <br>
  IVFPQ: Latency
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVFPQ_PQT.png"
    alt="IVFPQ: Per query time">
  <br>
  IVFPQ: Per query time
</p>

<p align="center">
  <img src="https://github.com/kshesha1/ANNBenchmark/blob/main/Graphs/Milvus-Index_operation/IVFPQ-QPS.png"
    alt="IVFPQ: QPS">
  <br>
  IVFPQ: QPS
</p>

