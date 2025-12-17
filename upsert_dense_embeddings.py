from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings
import os
import json
from typing import List, Iterable

# function that reads file in json line format, result is a tuple
# where (unique doc id, actual text) --> this is taken from the index bm42 python file from qdrant
def read_file(docs: str) -> Iterable[str]:
    data = []
    with open(docs, "r") as file:
        for line in file:
            row = json.loads(line)
            data.append((row["_id"], row["text"])) 
    return data

# to use Langchain API, insert your specific properties 
os.environ["OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_TYPE"] = "azure"

embedding_function = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)


def main():
    
    #insert path to dataset
    dataset = "./quora/corpus.jsonl"
    results = (read_file(dataset))
    ids = []
    texts = []
    for t in results:
        id = int(t[0])
        text = t[1]
        ids.append(id)
        texts.append(text)
    

    # This automatically generates Azure OpenAI embeddings and inserts them into Qdrant
    # specify existing collection
    qdrant = QdrantVectorStore.from_texts(texts=texts,
                                        embedding=embedding_function,
                                        ids = ids,
                                        vector_name="openai",
                                        url="http://localhost:6333",
                                        collection_name="quora_collection", batch_size = int(1000)
        )

if __name__=="__main__":
    main()
