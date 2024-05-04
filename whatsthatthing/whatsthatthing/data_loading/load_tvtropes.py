import pandas as pd
from pathlib import Path

input_path = Path(__file__).parents[3] / "input_data"


def load_print_data():
    df_tropes = pd.read_csv(input_path / "tropes.csv")
    df_tv_tropes = pd.read_csv(input_path / "tv_tropes.csv")
    print(df_tropes.columns)
    print(df_tropes["Trope"][:10])
    print(df_tropes["TropeID"][:10])
    #print(df_tropes.head(5))


if __name__ == "__main__":
    load_print_data()