import sqlite3
import pandas as pd

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
    return
    # es_player_attributes = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
    # # print(f"es_player_attributes -> {es_player_attributes}")
    # fifa_df = pd.read_csv("data/data.csv")
    # fifa_df = fifa_df.rename(columns={"ID":"player_fifa_api_id"})
    # # print(f'fifa_df -> {fifa_df}')
    #
    # mergeDf = es_player_attributes.merge(fifa_df[["player_fifa_api_id","Position"]])
    # print(f'mergeDf -> {mergeDf}')
    # mergeDf.to_sql("player_position",con =conn)

def main():
    database = r"data/database.sqlite"
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("Successful")
        cur = conn.cursor()
        player_position_df = pd.read_sql_query("SELECT * FROM player_position", conn)
        print(f"player_position_df -> {player_position_df}")

        player_position_df = player_position_df.dropna(axis=0, subset=['Position'])
        print(f"player_position_df size-> {player_position_df}")
        print(f"unique position values -> {player_position_df.Position.unique()}")
        print(f"null values -> {player_position_df.isna().sum()}")

if __name__ == "__main__":
    main()