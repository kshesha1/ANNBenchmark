import time
import tarfile

from hnsw import (
    connect_to_milvus,
    check_collection_exists,
    create_collection,
    insert_data,
    create_index,
    search_based_on_vector_similarity,
    measure_latency_and_pqt,
    measure_qps,
    load_data,
    measure_build_times,
    define_search_params
)

import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 128

def measure_build_times(collection, schema, wb, num_vectors):
    print("\n=== Build times ===")
    build_times = []
    for num_vector in num_vectors:
        start_time = time.time()
        vectors_to_build = [wb[:num_vector].tolist()]
        if has_collection:
            collection.drop()
        collection = Collection("sift1m_collection", schema)
        create_index(collection)  # Create index after re-creating the collection
        collection.insert(vectors_to_build)
        collection.flush()
        end_time = time.time()
        build_time = end_time - start_time
        build_times.append(build_time)
        print(f"Number of vectors: {num_vector}, Build Time: {build_time:.4f}s")
    print("=== Done with Build times ===")
    return build_times


if __name__ == "__main__":
    connect_to_milvus()
    has_collection = check_collection_exists("sift1m_collection")
    wb, xq = load_data()
    sift1m_collection = create_collection()
    schema = sift1m_collection.schema  # Get the schema for later use

    create_index(sift1m_collection)  # Create index first
    insert_data(sift1m_collection, wb)

    search_params = define_search_params()

    search_based_on_vector_similarity(sift1m_collection, xq[:10].tolist(), search_params)

    num_vectors = [10, 50, 100, 500, 1000]

    latencies, pqt_list = measure_latency_and_pqt(sift1m_collection, xq, num_vectors, search_params)
    qps_list = measure_qps(sift1m_collection, xq, num_vectors, search_params)

    build_times = measure_build_times(sift1m_collection, schema, wb, num_vectors)
