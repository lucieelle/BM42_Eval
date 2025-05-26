from fastembed import SparseTextEmbedding
import os
import json
from typing import List, Iterable, Tuple
from qdrant_client import QdrantClient, models
from tqdm import tqdm 
from qdrant_client.models import PointStruct, VectorParams, Distance
import math


def index_splade(documents: List[str]):
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    embeddings = list(model.embed(documents))
    return embeddings

def count_lines(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def stream_and_index_with_ids(filepath: str, batch_size: int = 15) -> List[Tuple[int, List[float]]]:
    total_docs = count_lines(filepath)
    total_batches = math.ceil(total_docs / batch_size)
    print(f"Total documents: {total_docs}")
    print(f"Total batches: {total_batches}")

    results = []
    batch_ids = []
    batch_texts = []
    current_batch = 1

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            doc_id = int(row["_id"])
            text = row["text"]

            batch_ids.append(doc_id)
            batch_texts.append(text)

            if len(batch_texts) == batch_size:
                print(f"Indexing batch {current_batch}/{total_batches}...")
                embeddings = index_splade(batch_texts)
                results.extend(zip(batch_ids, embeddings))
                batch_ids = []
                batch_texts = []
                current_batch += 1

        # Final batch
        if batch_texts:
            print(f"Indexing final batch {current_batch}/{total_batches}...")
            embeddings = index_splade(batch_texts)
            results.extend(zip(batch_ids, embeddings))

    return results

def upload_datapoints(embeddings, vector_name):
    points_list = []
    for doc_id, embedding in embeddings:
        # skip empty sparse vectors
        if len(embedding.values) == 0 or len(embedding.indices) == 0:
            print(f"Skipping doc_id {doc_id} due to empty vector")
            continue

        point = models.PointVectors(
            id=doc_id,
            vector={
                vector_name: models.SparseVector(
                    values=embedding.values.tolist(),
                    indices=embedding.indices.tolist()
                )
            },
        )
        points_list.append(point)
    return points_list



def main():
    VECTOR_NAME = "splade"
    COLLECTION_NAME = "quora_collection"
    BATCH_SIZE = 200
    
    client = QdrantClient(url="http://localhost:6333", timeout=300)
    dataset = "./corpus.jsonl"
    
    embeddings = stream_and_index_with_ids(dataset)
    points = upload_datapoints(embeddings, vector_name=VECTOR_NAME)


    for i in tqdm(range(0, len(points), BATCH_SIZE)):
        batch = points[i:i+BATCH_SIZE]

        try:
            client.update_vectors(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        except Exception as e:
            print(f"Batch {i}-{i+BATCH_SIZE} failed: {e}")


if __name__=="__main__":
    main()