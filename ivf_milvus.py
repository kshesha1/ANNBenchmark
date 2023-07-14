import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 128

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("sift1m_collection")
print(f"Does collection sift1m_collection exist in Milvus: {has}")
#################################################################################
# 2. create collection
import tarfile
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Extract the contents of the tar.gz file
tar = tarfile.open('sift.tar.gz', "r:gz")
tar.extractall()

# Define a function to read the fvecs file format of the Sift1M dataset
def read_fvecs(fp):
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

# Read the Sift1M dataset files
wb = read_fvecs('./sift/sift_base.fvecs')  # 1M samples
xq = read_fvecs('./sift/sift_query.fvecs')


# Define the dimensionality of the embeddings
dim = wb.shape[1]

# Create the collection schema
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
schema = CollectionSchema(fields, "Sift1M dataset collection")
print(fmt.format("Create collection `sift1m_collection`"))

has_collection = utility.has_collection("sift1m_collection")
print(has_collection)

# Create or load the collection
if has_collection:
    sift1m_collection = Collection("sift1m_collection")
    sift1m_collection.drop()  # Drop the existing collection

sift1m_collection = Collection("sift1m_collection", schema) # Create the collection



print(fmt.format("Start inserting entities"))
wb = wb[:10000]
# Prepare the entities for insertion
entities = [
    wb.tolist()  # Embeddings field
]

# Insert the entities into the collection
sift1m_collection.insert(entities)
sift1m_collection.flush()

# Check the number of entities in the collection
print(f"Number of entities in Milvus: {sift1m_collection.num_entities}")


# 4. create index
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

sift1m_collection.create_index("embeddings", index)

print("============Done with index===================")


# 5. search, query, and hybrid search

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
sift1m_collection.load()

# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
vectors_to_search = xq[:10].tolist()
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = sift1m_collection.search(vectors_to_search, "embeddings", search_params, limit=10, output_fields = [])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}")
print(search_latency_fmt.format(end_time - start_time))



print("============Done with search===================")


num_vectors = [10, 50, 100, 500, 1000]  # Number of vectors to search

latencies = []

for num in num_vectors:
    vectors_to_search = xq[:num].tolist()

    start_time = time.time()
    result = sift1m_collection.search(vectors_to_search, "embeddings", search_params, limit=10, output_fields=[])
    end_time = time.time()

    latency = end_time - start_time
    latencies.append(latency)

    print(f"Number of vectors: {num}, Latency: {latency:.4f}s")

'''import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=num_vectors, y=latencies, mode='lines'))
fig.update_layout(
    title="Latency vs Number of Vectors to Search",
    xaxis_title="Number of Vectors",
    yaxis_title="Latency (s)",
    showlegend=False
)
fig.show()'''

print("============Done with latency vs queries===================")

print("============QPS===================")


# Measure QPS
qps_list = []

for num_query in num_vectors:
    vectors_to_search = xq[:num_query].tolist()

    start_time = time.time()
    query_count = 0

    while time.time() - start_time < 1.0:  # Measure QPS for 1 second
        sift1m_collection.search(vectors_to_search, "embeddings", search_params, limit=10, output_fields=[])
        query_count += num_query

    qps = query_count / (time.time() - start_time)
    qps_list.append(qps)

    print(f"Number of vectors: {num_query}, QPS: {qps:.4f}")


'''import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=num_vectors, y=qps_list, mode='lines'))
fig.update_layout(
    title="QPS vs Number of Queries",
    xaxis_title="Number of Queries",
    yaxis_title="Queries per Second (QPS)",
    showlegend=False
)
fig.show()'''

print("============Done with QPS===================")

print("============Build times===================")
#num_vectors = [100, 500, 1000]  # Number of vectors for build time measurement

build_times = []

for num_vector in num_vectors:
    #entities = [wb.tolist()]  # Embeddings field
    
    vectors_to_build = [wb[:num_vector].tolist()]

    start_time = time.time()
    sift1m_collection.insert(vectors_to_build)
    sift1m_collection.flush()
    end_time = time.time()

    build_time = end_time - start_time
    build_times.append(build_time)

    print(f"Number of vectors: {num_vector}, Build Time: {build_time:.4f}s")

print("============done with Build times===================")
