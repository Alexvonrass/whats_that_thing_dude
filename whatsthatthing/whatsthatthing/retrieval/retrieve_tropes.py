import pdb

import pandas as pd
from pathlib import Path
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

db_path = Path(__file__).parents[3] / "db" / "trope_db"


def retrieve_tropes(prompt: str):
    embedding_model = GPT4AllEmbeddings()
    vector_store = Chroma(persist_directory=str(db_path), embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke(prompt)
    a = retrieved_docs[0]
    print(a.lc_id())
    print(a.__dir__())
    print(retrieved_docs[0])


if __name__ == "__main__":
    retrieve_tropes("That thing when you do a ton of preparation for a comically small payoff")