import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np

from ai_framework.VectorDBHandler import VectorDBHandler


class PostgresDBHandler(VectorDBHandler):
    """
    A class that handles the connection and operations with a PostgreSQL database.

    Attributes:
        configuration_db (dict): A dictionary containing the configuration for the PostgreSQL database.
        verbose_level (int): The level of verbosity for the class.
    """

    def __init__(self, configuration_db, verbose_level=1):
        super().__init__(configuration_db, verbose_level=verbose_level)

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        self.close()
        # Establishing the connection
        self.conn = psycopg2.connect(
            database=self.configuration_db["postgresql"]["database"],
            user=self.configuration_db["postgresql"]["user"],
            password=self.configuration_db["postgresql"]["password"],
            host=self.configuration_db["postgresql"]["host"],
            port=self.configuration_db["postgresql"]["port"],
        )

    def execute_statement(self, statement, commit=True, open_and_close=False):
        """
        Executes a SQL statement in the database.

        Parameters:
            statement (str): The SQL statement to be executed.
            commit (bool): Whether to commit the changes after execution.
            open_and_close (bool): Whether to open and close the connection automatically.

        Returns:
            cursor: The cursor object after executing the statement.
        """
        self._check_open(open_and_close)

        cursor = self.conn.cursor()
        cursor.execute(statement)

        self._check_commit_and_close(commit, open_and_close)
        return cursor

    def enable_vector_extension(self):
        """Enables the vector extension in the PostgreSQL database."""
        self.execute_statement(
            "CREATE EXTENSION IF NOT EXISTS vector", open_and_close=True
        )

    def register_vector_in_conn(self):
        """Registers the vector extension in the current connection."""
        register_vector(self.conn)

    def insert_values(
        self,
        data_list,
        sql_columns,
        register_vector_conn=True,
        commit=True,
        open_and_close=False,
    ):
        """
        Inserts a list of values into the database.

        Parameters:
            data_list (list): A list of values to be inserted.
            sql_columns (str): The SQL columns to insert the values into.
            register_vector_conn (bool): Whether to register the vector extension in the connection.
            commit (bool): Whether to commit the changes after insertion.
            open_and_close (bool): Whether to open and close the connection automatically.
        """
        self._check_open(open_and_close)

        if register_vector_conn:
            self.register_vector_in_conn()

        cursor = self.conn.cursor()

        execute_values(cursor, f"INSERT INTO {sql_columns} VALUES %s", data_list)

        self._check_commit_and_close(commit, open_and_close)

    def get_similar_docs(
        self,
        embedder,
        user_input,
        schema,
        table_name,
        k=3,
        custom_where="",
        columns="content",
        register_vector_conn=True,
        commit=True,
        open_and_close=False,
        col_name_embedding="embedding",
        partition_by=None,
    ):
        """
        Retrieves the top k most similar documents to a given user input based on the defined index.

        Parameters:
            embedder (Embedder): An instance of the Embedder class with its configuration.
            user_input (str): The user's query string.
            schema (str): The PostgreSQL schema where the table with the embeddings is located.
            table_name (str): The name of the table with the embeddings.
            k (int): The number of documents to retrieve.
            custom_where (str): An optional custom WHERE clause to filter the data in the embeddings table.
            columns (str): A comma-separated string of columns to retrieve.
            register_vector_conn (bool): Whether to register the vector extension in the connection.
            commit (bool): Whether to commit the changes after the operation.
            open_and_close (bool): Whether to open and close the connection automatically.
            col_name_embedding (str): The name of the column that contains the embeddings.
            partition_by (str|None): The column to partition the results by. Set to None to disable partitioning.

        Returns:
            top_docs (List[Tuple]): A list of tuples containing the top k most similar documents and their distances.
        """

        self._check_open(open_and_close)

        if register_vector_conn:
            self.register_vector_in_conn()

        self.embedding_array = np.array(embedder.encoder.embed_query(user_input))

        cur = self.conn.cursor()

        if partition_by is None:
            sql_pg = f"""SELECT {columns}, {col_name_embedding} <=> %s as distance
            FROM {schema}.{table_name} {custom_where}
            ORDER BY distance ASC LIMIT {k}
            """

            args = (self.embedding_array,)
        else:
            sql_pg = f"""SELECT
            {columns},
            distance
            FROM (
                SELECT {columns}, {col_name_embedding} <=> %s as distance,
                ROW_NUMBER() OVER(PARTITION BY {partition_by} ORDER BY {col_name_embedding} <=> %s ASC) AS pos
                FROM {schema}.{table_name} {custom_where}
            ) as temp
            WHERE pos = 1
            ORDER BY distance ASC LIMIT {k}
            """
            args = (self.embedding_array, self.embedding_array)

        # Get the top k most similar documents using the KNN <=> operator
        cur.execute(
            sql_pg,
            args,
        )

        top_docs = cur.fetchall()

        self._check_commit_and_close(commit, open_and_close)

        return top_docs

    def recreate_index(self, params):
        """
        Regenerates the index after a while.

        Raises:
            Exception: This method is not implemented.
        """
        raise Exception("recreate_index: NOT IMPLEMENTED")
