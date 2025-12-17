from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)              

# this function updates existing collection's parameters
def update_collection(collection_name: str):
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(),
        )
    return True
# this function creates a new collection with thesis specific configuration
def create_qdrant_collection(collection_name: str):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "openai": models.VectorParams(size=1536, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "bm42": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "splade": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )
    return True
   

def main():
    # change collection name
    collection_name = "quora_collection"

    create_qdrant_collection(collection_name)
    info = client.get_collection(collection_name=collection_name)
    print(f"Created collection - {info}")

if __name__=="__main__":
    main()

