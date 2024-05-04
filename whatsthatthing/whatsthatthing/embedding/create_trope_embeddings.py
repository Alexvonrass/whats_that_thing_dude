import pandas as pd
from pathlib import Path
from langchain import hub
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

# Set the input path
input_path = Path(__file__).parents[3] / "input_data"
db_path = Path(__file__).parents[3] / "db" / "trope_db"

def load_print_data():
    # Load data
    df_tropes = pd.read_csv(input_path / "tropes.csv")
    df_tropes.rename(columns={"TropeID": "trope_id",
                              "Description": "description",
                              "Trope": "trope_name"}, inplace=True)
    #df_tropes["trope_id"] = df_tropes["trope_id"].fillna("No description")
    df_tropes["description"] = df_tropes["description"].fillna("No description")
    df_tropes["name_and_description"] = df_tropes["trope_name"] + ":" + df_tropes["description"]
    df_tropes = df_tropes.sample(n=1000).copy()

    # Initialize the local embeddings model
    embedding_model = GPT4AllEmbeddings()

    # Initialize Chroma VectorStore
    vector_store = Chroma.from_texts(texts=df_tropes["name_and_description"].tolist(),
                                     ids=df_tropes["trope_id"].tolist(),
                                     embedding=embedding_model,
                                     persist_directory=str(db_path))
    return None


if __name__ == "__main__":
    load_print_data()