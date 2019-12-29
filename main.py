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


def main():
    database = r"data/database.sqlite"
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("Successful")
        cur = conn.cursor()
        df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
        print(df["positioning"])
        # iterating the columns
        # for col in df.columns:
        #     print(col)


if __name__ == "__main__":
    main()