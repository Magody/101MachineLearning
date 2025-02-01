from abc import ABC, abstractmethod
from ai_framework.LoggingAndTelemetry import EnumLogs
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import numpy as np


class VectorDBHandler(ABC):
    """
    An abstract base class that provides a common interface for handling vector databases.

    Attributes:
        configuration_db (object): The configuration object for the database.
        conn (object): The connection object for the database.
        verbose_level (EnumLogs): The verbosity level for logging.
        verbose (bool): A flag indicating whether verbose logging is enabled.
        is_mock (bool): A flag indicating whether the handler is in mock mode.
        schema_base (StructType): The base schema for the vector database.
    """

    def __init__(
        self,
        configuration_db,
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        is_mock: bool = False,
        spark=None
    ):
        self.configuration_db = configuration_db
        self.conn = None
        self.verbose_level = verbose_level
        self.verbose = self.verbose_level.value >= 1
        self.is_mock = is_mock
        self.schema_base = StructType(
            [
                StructField("id", StringType(), True),
                StructField("contenido_original", StringType(), True),
                StructField("contenido_vector", ArrayType(FloatType()), True),
                StructField("vectorizacion_algoritmo", StringType(), True),
                StructField("version", StringType(), True),
                StructField("fecha_actualizacion", StringType(), True),
                StructField("usuario_actualizacion", StringType(), True),
                StructField("fecha_creacion", StringType(), True),
                StructField("usuario_creacion", StringType(), True),
            ]
        )
        self.spark = spark

        super().__init__()

    def close(self):
        """
        Establishes a connection to the database.
        This method must be implemented by subclasses.
        """
        # No abstracto pero si acepta override si close() no existe
        if self.conn is not None:
            self.conn.close()

        self.conn = None

    @abstractmethod
    def connect(self):
        # Abstracto, debe implementarse obligatoriamente
        self.close()
        # Establishing the connection
        # self.conn = <CONNECTION LOGIC>

    def __del__(self):
        self.close()

    def _check_open(self, open_and_close):
        """
        Checks if the database connection is open and establishes it if necessary.

        Args:
            open_and_close (bool): A flag indicating whether to open and close the connection.
        """
        if open_and_close:
            self.connect()
        else:
            if self.conn is None and not self.is_mock:
                raise Exception(
                    """Create connection first
                    (use self.connect() and other
                    functions needed for startup)
                    """
                )

    def _check_commit_and_close(self, commit=True, open_and_close=False):
        """
        Commits the current transaction and closes the database connection if necessary.

        Args:
            commit (bool): A flag indicating whether to commit the transaction.
            open_and_close (bool): A flag indicating whether to open and close the connection.
        """
        if self.conn is not None and not self.is_mock:
            if commit:
                self.conn.commit()

        if open_and_close:
            self.close()

    @abstractmethod
    def recreate_index(self, params):
        """
        Recreates the index in the vector database.
        This method must be implemented by subclasses.

        Args:
            params (object): The parameters for recreating the index.
        """
        pass

    @abstractmethod
    def insert_values(
        self,
        columns,
        data_list,
        commit=True,
        open_and_close=False,
    ):
        """
        Inserts values into the vector database.
        This method must be implemented by subclasses.

        Args:
            columns (list): The columns to insert the values into.
            data_list (list): The list of data to insert.
            commit (bool): A flag indicating whether to commit the transaction.
            open_and_close (bool): A flag indicating whether to open and close the connection.
        """
        self._check_open(open_and_close)

        # Create a cursor and save it

        self._check_commit_and_close(commit, open_and_close)

    @abstractmethod
    def get_similar_docs(
        self,
        embedder,  # :Embedder,
        user_input,
        schema,
        table_name,
        k=3,
        custom_where="",
        columns="",
        commit=True,
        open_and_close=False,
        col_name_embedding="embedding",
        partition_by=None,
    ):
        """
        Retrieves the top k most similar documents to a given user input.

        Parameters:
            embedder (Embedder): An instance of the Embedder class with its configuration.
            user_input (str): The user's query.
            schema (str): The schema or additional information where the embedding table is located.
            table_name (str): The name of the table containing the embeddings.
            k (int): The number of documents to retrieve.
            custom_where (str): An optional custom WHERE condition to filter the embedding table.
            columns (str): A comma-separated list of columns to retrieve.
            commit (bool): A flag indicating whether to commit the transaction.
            open_and_close (bool): A flag indicating whether to open and close the connection.
            col_name_embedding (str): The name of the column containing the embeddings.
            partition_by (str|None): The column to partition by, if desired. Set to None otherwise.

        Returns:
            top_docs (List[Tuple]): A list of tuples containing the top k most similar documents and their distances.
        """

        self._check_open(open_and_close)

        self.embedding_array = np.array(embedder.encoder.embed_query(user_input))

        top_docs = [
            (),
        ]

        self._check_commit_and_close(commit, open_and_close)

        return top_docs
