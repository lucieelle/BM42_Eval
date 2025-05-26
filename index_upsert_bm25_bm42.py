from fastembed import SparseTextEmbedding
import json
from typing import List, Iterable, Tuple
from qdrant_client import QdrantClient, models
from tqdm import tqdm 
from qdrant_client.models import PointStruct, VectorParams, Distance

# function that reads file in json line format, result is a tuple where (unique doc id, actual text) --> this is taken from the index bm42 python file from qdrant

def read_file(docs: str) -> Iterable[str]:
    data = []
    with open(docs, "r") as file:
        for line in file:
            row = json.loads(line)
            data.append((row["_id"], row["text"])) 
    return data

def index_bm25(documents):
    model = SparseTextEmbedding(model_name="Qdrant/bm25", avg_len =  181)
    embeddings = list(model.embed(documents))
    return embeddings

def index_bm42(documents):
    model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")    
    embeddings = list(model.embed(documents))
    return embeddings

def upload_datapoints(ids, embeddings, vector_name):
    points_list = []
    for doc_id, embedding in zip(ids, embeddings):
        # Safely skip empty sparse vectors
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
    VECTOR_NAME = "bm25"
    dataset = "./corpus.jsonl"
    COLLECTION_NAME = "quora_collection"


    documents = (read_file(dataset))

    ids = []
    texts = []
    for t in documents:
        id = int(t[0])
        text = t[1]
        ids.append(id)
        texts.append(text)

    client = QdrantClient(url="http://localhost:6333", timeout=300)

    if VECTOR_NAME == "bm25":
        print("imbedding with bm25")
        embeddings = index_bm25(texts)
    else:
        print("imbedding with bm42")
        embeddings = index_bm42(texts)
    print("Idexing with", VECTOR_NAME)

    points = upload_datapoints(ids, embeddings, vector_name=VECTOR_NAME)


    BATCH_SIZE = 100

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
