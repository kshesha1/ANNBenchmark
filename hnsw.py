import time
import tarfile
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Constants
fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 128


def connect_to_milvus():
    print("Start connecting to Milvus")
    connections.connect("default", host="localhost", port="19530")

def check_collection_exists(collection_name):
    return utility.has_collection(collection_name)

def create_collection():
    print("Create collection `sift1m_collection`")
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "Sift1M dataset collection")

    # Drop the existing collection if it exists
    has_collection = check_collection_exists("sift1m_collection")
    if has_collection:
        sift1m_collection = Collection("sift1m_collection")
        sift1m_collection.drop()

    # Create the collection
    sift1m_collection = Collection("sift1m_collection", schema)


    return sift1m_collection

def load_data():
    tar = tarfile.open('sift.tar.gz', "r:gz")
    tar.extractall()

    def read_fvecs(fp):
        a = np.fromfile(fp, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

    wb = read_fvecs('./sift/sift_base.fvecs')  # 1M samples
    xq = read_fvecs('./sift/sift_query.fvecs')
    return wb, xq

def insert_data(collection, wb):
    print("Start inserting entities")
    wb = wb[:10000]
    entities = [
        wb.tolist()  # Embeddings field
    ]
    collection.insert(entities)
    collection.flush()

def create_index(collection):
    print("Start Creating index HNSW")
    index = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 64},
    }
    collection.create_index("embeddings", index)

def define_search_params():
    return {
        "metric_type": "L2",
        "params": {"ef": 64},
    }

def search_based_on_vector_similarity(collection, xq_subset, search_params):
    print("Start loading")
    collection.load()

    print("Start searching based on vector similarity")
    start_time = time.time()
    result = collection.search(xq_subset, "embeddings", search_params, limit=10, output_fields=[])
    end_time = time.time()

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}")
    print(search_latency_fmt.format(end_time - start_time))

def measure_latency_and_pqt(collection, xq, num_vectors, search_params):
    latencies = []
    pqt_list = []

    for num in num_vectors:
        vectors_to_search = xq[:num].tolist()

        start_time = time.time()
        result = collection.search(vectors_to_search, "embeddings", search_params, limit=10, output_fields=[])
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)
        pqt = latency / num
        pqt_list.append(pqt)

        print(f"Number of vectors: {num}, Latency: {latency:.4f}s, PQT: {pqt:.4f}")

    return latencies, pqt_list

def measure_qps(collection, xq, num_vectors, search_params):
    qps_list = []

    for num in num_vectors:
        vectors_to_search = xq[:num].tolist()

        start_time = time.time()
        query_count = 0

        while time.time() - start_time < 1.0:  # Measure QPS for 1 second
            collection.search(vectors_to_search, "embeddings", search_params, limit=10, output_fields=[])
            query_count += num

        qps = query_count / (time.time() - start_time)
        qps_list.append(qps)
        print(f"Number of vectors: {num}, QPS: {qps:.4f}")

    return qps_list

def measure_build_times(collection, schema, wb, num_vectors):
    build_times = []
    has_collection = check_collection_exists("sift1m_collection")

    for num_vector in num_vectors:
        start_time = time.time()
        vectors_to_build = [wb[:num_vector].tolist()]

        if has_collection:
            collection.drop()
        collection = Collection("sift1m_collection", schema)

        collection.insert(vectors_to_build)
        collection.flush()

        create_index(collection)

        end_time = time.time()

        build_time = end_time - start_time
        build_times.append(build_time)

        print(f"Number of vectors: {num_vector}, Build Time: {build_time:.4f}s")

    return build_times

def main():
    connect_to_milvus()
    wb, xq = load_data()
    sift1m_collection = create_collection()
    schema = sift1m_collection.schema  # Get the schema for later use

    insert_data(sift1m_collection, wb)
    create_index(sift1m_collection)
    search_params = define_search_params()

    search_based_on_vector_similarity(sift1m_collection, xq[:10].tolist(), search_params)

    num_vectors = [10, 50, 100, 500, 1000]

    latencies, pqt_list = measure_latency_and_pqt(sift1m_collection, xq, num_vectors, search_params)
    qps_list = measure_qps(sift1m_collection, xq, num_vectors, search_params)

    build_times = measure_build_times(sift1m_collection, schema, wb, num_vectors)

if __name__ == "__main__":
    main()
