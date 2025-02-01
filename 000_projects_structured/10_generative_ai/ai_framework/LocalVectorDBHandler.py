import numpy as np
from ai_framework.VectorDBHandler import VectorDBHandler
import nmslib


class LocalVectorDBHandler(VectorDBHandler):
    """
    A class that inherits from the VectorDBHandler class and provides
    functionality for connecting to a local vector database, inserting
    values, recreating the index, and retrieving similar documents.

    Attributes:
        configuration_db (dict): A dictionary containing the configuration
            for the database connection.
        verbose_level (int): The level of verbosity for logging.
        is_mock (bool): Indicates whether the handler is a mock implementation.
        df (pyspark.sql.DataFrame): The DataFrame containing the data.
        index (nmslib.Index): The HNSW index used for similarity search.
        embedding_array (numpy.ndarray): The embedding array for the user input.

    Methods:
        connect(option='sample', extra={}): Connects to the local vector database
            and initializes the HNSW index.
        recreate_index(params): Recreates the HNSW index with the specified parameters.
        execute_statement(statement, commit=True, open_and_close=False): Executes a
            SQL statement on the DataFrame.
        insert_values(data_list, sql_columns, register_vector_conn=True, commit=True,
            open_and_close=False): Inserts values into the HNSW index.
        get_similar_docs(embedder, user_input, schema, table_name, k=3, custom_where="",
            columns="content", commit=True, open_and_close=False, col_name_embedding="embedding",
            partition_by=None): Retrieves the top k similar documents to the user input.

    Use:
    from dotenv import load_dotenv
    import os
    from ai_framework.VectorDBHandler import VectorDBHandler
    from ai_framework.LocalVectorDBHandler import LocalVectorDBHandler

    path_dotenv = "/Workspace/Users/ddiazpad@pichincha.com/CelulaIA/.env"
    load_dotenv(path_dotenv)

    vector_local_handler:VectorDBHandler = LocalVectorDBHandler({}, verbose_level=EnumLogs.LOG_LEVEL_DEBUG)
    vector_local_handler.connect()

    query_vector = list(map(float, np.random.rand(1536)))
    ids, distances = vector_local_handler.index.knnQuery(query_vector, k=3)

    df_document_query = spark.createDataFrame(
        [(int(ids[i]), float(distances[i])) for i in range(len(ids))],
        schema="id_vector int, distance float"
    )

    vector_local_handler.df.join(
        df_document_query,
        vector_local_handler.df["id"] == df_document_query["id_vector"],
        "left"
    ).filter("distance is not NULL").display()
    """

    def __init__(self, configuration_db, spark, verbose_level=1):

        super().__init__(configuration_db, verbose_level=verbose_level, is_mock=True, spark=spark)

    def connect(self, option="sample", extra={}):
        """Simulates a connection to a sample inner data"""
        self.close()
        if option == "sample":
            embedder = extra.get("embedder", None)
            amount = 10

            contenido_original = [f"dummy relacionado a {i}" for i in range(amount)]

            if embedder is not None:
                contenido_vector = [
                    embedder.encoder.embed_query(contenido)
                    for contenido in contenido_original
                ]
            else:
                contenido_vector = [
                    list(map(float, np.random.rand(1536))) for _ in range(amount)
                ]

            data = {
                "id": [i for i in range(amount)],
                "contenido_original": contenido_original,
                "contenido_vector": contenido_vector,
                "vectorizacion_algoritmo": [None for _ in range(amount)],
                "version": ["1" for _ in range(amount)],
                "fecha_actualizacion": ["2024-02-20" for _ in range(amount)],
                "usuario_actualizacion": [
                    "ddiazpad@pichincha.com" for _ in range(amount)
                ],
                "fecha_creacion": ["2024-02-20" for _ in range(amount)],
                "usuario_creacion": ["ddiazpad@pichincha.com" for _ in range(amount)],
            }
            data_tuples = list(zip(*data.values()))
            self.df = self.spark.createDataFrame(data_tuples, schema=self.schema_base)

            df_pandas = self.df.select(
                "id", "contenido_original", "contenido_vector"
            ).toPandas()

            # Inicializar un índice HNSW
            self.index = nmslib.init(method="hnsw", space="cosinesimil")
            self.insert_values(df_pandas, None)

            # Crear el índice
            self.recreate_index(
                params={"input_function": {"post": 2}, "print_progress": True}
            )

            return self.df

        else:
            raise Exception("connect: NOT IMPLEMENTED")

    def recreate_index(self, params):
        """
        Recreates the HNSW index with the specified parameters.

        Parameters:
            params (dict): A dictionary containing the parameters for recreating the index.
                - input_function (dict): A dictionary with the 'post' key, which specifies the
                  post-processing function for the input data.
                - print_progress (bool): Indicates whether to print the progress of the index
                  creation.
        """
        self.index.createIndex(
            params["input_function"], print_progress=params["print_progress"]
        )

    def execute_statement(self, statement, commit=True, open_and_close=False):
        """
        Executes a SQL statement on the DataFrame.

        Parameters:
            statement (str): The SQL statement to execute.
            commit (bool): Indicates whether to commit the changes after execution.
            open_and_close (bool): Indicates whether to open and close the connection
                before and after execution.

        Returns:
            cursor: The cursor object returned by the database execution.
        """
        self._check_open(open_and_close)
        pass
        self._check_commit_and_close(commit, open_and_close)
        return None  # cursor

    def insert_values(
        self,
        data_list,
        sql_columns,
        register_vector_conn=True,
        commit=True,
        open_and_close=False,
    ):
        """
        Inserts values into the HNSW index.

        Parameters:
            data_list (pandas.DataFrame): The DataFrame containing the data to be inserted.
            sql_columns (list): A list of column names to be inserted.
            register_vector_conn (bool): Indicates whether to register the vector connection.
            commit (bool): Indicates whether to commit the changes after insertion.
            open_and_close (bool): Indicates whether to open and close the connection
                before and after insertion.
        """
        self._check_open(open_and_close)

        for _, row in data_list.iterrows():
            self.index.addDataPoint(int(row["id"]), row["contenido_vector"])

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
        commit=True,
        open_and_close=False,
        col_name_embedding="embedding",
        partition_by=None,
    ):
        """
        Retrieves the top k similar documents to the user input.

        Parameters:
            embedder (Embedder): The Embedder object used to generate the embeddings.
            user_input (str): The user input for which to find similar documents.
            schema (str): The schema of the table containing the embeddings.
            table_name (str): The name of the table containing the embeddings.
            k (int): The number of similar documents to retrieve.
            custom_where (str): An optional custom WHERE clause to filter the data.
            columns (str): A comma-separated list of columns to retrieve.
            commit (bool): Indicates whether to commit the changes after the operation.
            open_and_close (bool): Indicates whether to open and close the connection
                before and after the operation.
            col_name_embedding (str): The name of the column containing the embeddings.
            partition_by (str|None): The column to partition the results by. Set to None
                to return all results.

        Returns:
            top_docs (List[Tuple]): A list of tuples containing the IDs and distances
                of the top k similar documents.
        """

        self._check_open(open_and_close)

        self.embedding_array = np.array(embedder.encoder.embed_query(user_input))

        ids, distances = self.index.knnQuery(self.embedding_array, k=k)

        self._check_commit_and_close(commit, open_and_close)

        return [(ids[i], distances[i]) for i in range(len(ids))]
