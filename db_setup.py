import json
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.docstore.document import Document


def load_json_files(folder_path):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    file_data = json.load(file)
                    data.extend(file_data)
    return data


def split_text(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []

    for item in data:
        problem = item["problem"]
        solution = item["solution"]
        chunks = text_splitter.split_text(problem)
        documents.extend(
            [
                Document(page_content=chunk, metadata={"solution": solution})
                for chunk in chunks
            ]
        )

    return documents


data = load_json_files("merged_dataset/train")
if not data:
    print("No data loaded.")
else:
    documents = split_text(data)
    print("Number of documents:", len(documents))

    # Create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the data into Chroma
    db = Chroma.from_documents(
        documents, embedding=embedding_function, persist_directory="./chroma_db"
    )

    db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    # Query the vector store
    query = "What is the formula for the area of a circle?"
    docs = db2.similarity_search(query)

    # Print the results
    for doc in docs:
        print(doc.page_content)
        print("Solution:", doc.metadata["solution"])
        print("---")
