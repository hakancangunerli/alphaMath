import json
import os
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.docstore.document import Document
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


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

def test_rag():
    test_data = load_json_files("merged_dataset/test")

    if not test_data:
        raise Exception("No train data loaded.")
    else:
        #Load the data into Chroma
        db = get_rag()
        questions = []
        answers = []
        ground_truths = []
        contexts = []
        
        for question in test_data[:9]:

            # query = "What is the formula for the area of a circle?"
            query = question["problem"]
            solution = question["solution"]

            questions.append(query)
            ground_truths.append(solution)

            docs = db.similarity_search(query)

            answers.append(docs[0].metadata["solution"])
            contexts.append([doc.page_content for doc in docs])

        final_data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts
        }

        dataset = Dataset.from_dict(final_data)
        result = evaluate(
            dataset = dataset, 
            metrics=[
                context_precision,
                # context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        result.to_pandas().to_csv("rag_results.csv")

def load_rag():
    data = load_json_files("merged_dataset/train")
    if not data:
        raise Exception("No train data loaded.")
    else:
        documents = split_text(data)
        print("Number of documents:", len(documents))

        # Create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            #Load the data into Chroma
            db = Chroma.from_documents(
                documents, embedding=embedding_function, persist_directory="./chroma_db"
            )
            # query = data[0]["problem"]
            # solution = data[0]["solution"]
            # print("Original solution:", solution)
            # docs = db.similarity_search(query)
            # # Print the results
            # for doc in docs:
            #     print(doc.page_content)
            #     print("Solution:", doc.metadata["solution"])
            #     print("---")
        except:
            raise Exception("Rag Load Corrupted, please reload Rag")

def get_rag():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        #Load the data into Chroma
        db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        return db
    except:
        raise Exception("Rag Load Corrupted, please reload Rag")

# load_rag()
# test_rag()