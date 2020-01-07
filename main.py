import sqlite3
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Plotting
import numpy as np
import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS


from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

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
    player_position_df = pd.read_csv("data/all_float_data.csv")
    player_position_df = player_position_df.dropna(axis=0, subset=['Position'])

    print(f"player_position_df -> {player_position_df}")
    player_position_df = rearrange_columns(player_position_df)
    print(f'features -> {player_position_df}')



    # one_hot_encode(player_position_df["Position"].tolist())

    # plot_attr_distribution(player_position_df,"graphs")
    # player_position_df = convert_obj_float(player_position_df)
    # player_position_df = drop_unwanted_columns(player_position_df)
    # data_stats(player_position_df=player_position_df)
    # impute_object_columns(player_position_df)
    # impute_float_columns(player_position_df)
    # player_position_df.to_csv("data/all_float_data.csv",index=False)

def one_hot_encode(data):
    # data = player_position_df["Position"].tolist()
    values = np.array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(len(onehot_encoded[0]))
    # # invert first example
    # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    # print(inverted)

    # onehot_encoded_df = pd.DataFrame({"Position":onehot_encoded})
    # player_position_df.update(onehot_encoded_df)
    # print(f'features -> {player_position_df.Position}')
    return onehot_encoded

def convert_obj_float(player_position_df):
    player_position_df = player_position_df.replace({"preferred_foot": {"right": 1.0, "left": 2.0}})

    unwanted_column_values = ['1', '_0', '0', '2', '3', 'o', '7', 'ormal', '6', '9', '5',
                              '4']
    player_position_df = player_position_df[~player_position_df.defensive_work_rate.isin(unwanted_column_values)]
    player_position_df = player_position_df.replace({"defensive_work_rate": {"low": 1.0, "medium": 2.0, "high": 3.0}})

    player_position_df = player_position_df[~player_position_df.attacking_work_rate.isin(["None"])]
    player_position_df = player_position_df.replace({"attacking_work_rate": {"low": 1.0, "medium": 2.0, "high": 3.0}})
    return player_position_df

def rearrange_columns(player_position_df):
    rearranged_columns = [

        # Attacking attributes
        'attacking_work_rate',
        'crossing', 'finishing',
        'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
        'long_passing', 'ball_control', 'acceleration', 'sprint_speed','agility', 'reactions',  'shot_power',
        'long_shots', 'penalties',

        # Defensive attributes
        'defensive_work_rate', 'jumping', 'strength', 'interceptions', 'marking',
        'stamina', 'vision',
        'standing_tackle', 'sliding_tackle', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes',

        # Mixed attributes
        'overall_rating', 'aggression', 'balance', 'positioning','heading_accuracy','short_passing', 'potential',
        'preferred_foot', 'Position']
    return player_position_df[rearranged_columns]

def plot_attr_distribution(player_position_df,graph_directory):
    fig, ax = plt.subplots()
    df_new_ST = player_position_df[player_position_df['Position'] == 'ST'].iloc[::200, :-1]
    df_new_ST = df_new_ST.select_dtypes(float)
    cols = [col for col in df_new_ST.columns if col not in ['Position']]
    df_new_ST = df_new_ST.div(df_new_ST.sum(axis=1), axis=0)
    print(f'Final df plotted-> {df_new_ST}')
    df_new_ST.T.plot.line(color='black', figsize=(15, 15), legend=False, ylim=(0, 0.08),
                          title="ST's attributes distribution", ax=ax)

    ax.set_xlabel('Attributes')
    ax.set_ylabel('Rating')

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels=cols, rotation=90)

    for ln in ax.lines:
        ln.set_linewidth(1)

    ax.axvline(0, color='red', linestyle='--')
    ax.axvline(15, color='red', linestyle='--')

    ax.axvline(16, color='blue', linestyle='--')
    ax.axvline(29, color='blue', linestyle='--')

    ax.axvline(30, color='green', linestyle='--')
    ax.axvline(51, color='green', linestyle='--')

    ax.text(5, 100, 'Attack Attributes', color='red', weight='bold')
    ax.text(17, 100, 'Defend Attributes', color='blue', weight='bold')
    ax.text(33, 100, 'Mixed Attributes', color='green', weight='bold')

    # plt.savefig(graph_directory + "/line_plot_after_norm.png")
    plt.show()


def drop_unwanted_columns(player_position_df):
    drop_list = ["index","id",'player_fifa_api_id', 'player_api_id','date',]
    player_position_df = player_position_df.drop(drop_list,axis=1)
    print(f"player_position_df -> {player_position_df}")
    return player_position_df


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