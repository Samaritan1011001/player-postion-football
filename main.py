import sqlite3
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Plotting

import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return conn


def collect():
    database = r"data/database.sqlite"
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("Successful")
        cur = conn.cursor()
        es_player_attributes = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
        fifa_df = pd.read_csv("data/data.csv")
        fifa_df = fifa_df.rename(columns={"ID":"player_fifa_api_id"})
        mergeDf = es_player_attributes.merge(fifa_df[["player_fifa_api_id","Position"]])
        print(f'mergeDf -> {mergeDf}')
        mergeDf.to_sql("player_position",con =conn,if_exists='replace')
        # player_position_df = pd.read_sql_query("SELECT * FROM player_position", conn)
    return


def main():
    # collect(conn)
    player_position_df = pd.read_csv("data/final_data.csv")
    player_position_df = player_position_df.dropna(axis=0, subset=['Position'])
    print(f"player_position_df -> {player_position_df}")
    print(f'features -> {player_position_df.columns}')
    drop_list = ["index","id",'player_fifa_api_id', 'player_api_id','date',]
    player_position_df = player_position_df.drop(drop_list,axis=1)
    print(f"player_position_df -> {player_position_df}")

    # data_stats(player_position_df=player_position_df)
    # impute_object_columns(player_position_df)
    # impute_float_columns(player_position_df)
    # player_position_df.to_csv("data/no_ids_data.csv",index=False)


def data_stats(player_position_df):
    print(f"unique position values -> {player_position_df.Position.unique()}")
    print(f"null values -> {player_position_df.isna().sum()}")
    print(f"null value rows -> {player_position_df[player_position_df.isnull().any(axis=1)]}")
    # get_heatmap(player_position_df,"graphs")
    print(f"data types -> {player_position_df.dtypes}")
    print(f"Imputed player_position_df -> {player_position_df.isna().sum()}")


def impute_object_columns(player_position_df):
    player_position_df =  player_position_df.apply(lambda x: x.fillna(x.mode().iloc[0]))
    return player_position_df


def impute_float_columns(player_position_df):
    imputed_values = fast_knn(player_position_df.select_dtypes(float).values, k=30)
    float_df = pd.DataFrame(imputed_values,columns=list(player_position_df.select_dtypes('float64').columns))
    player_position_df.update(float_df)
    return player_position_df


def get_heatmap(df,graph_directory):
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.tight_layout()
    plt.savefig(graph_directory  +"/heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()