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


def define_search_params():
    return {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }


def create_index(collection):
    print(fmt.format("Start Creating index IVF_FLAT"))
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("embeddings", index)

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
