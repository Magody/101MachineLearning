import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np


class PostgresHandler:
    def __init__(self, configuration_db, verbose=1):
        self.configuration_db = configuration_db
        self.conn = None
        self.verbose = verbose

    def log(self, message, minimum_verbose=1):
        if self.verbose >= minimum_verbose:
            message_len = len(message)

            if message_len > 2000:
                print(message[0:2000])
            else:
                print(message)

    def connect(self):
        self.close()
        # Establishing the connection
        self.conn = psycopg2.connect(
            database=self.configuration_db["postgresql"]["database"],
            user=self.configuration_db["postgresql"]["user"],
            password=self.configuration_db["postgresql"]["password"],
            host=self.configuration_db["postgresql"]["host"],
            port=self.configuration_db["postgresql"]["port"],
        )

    def close(self):
        if self.conn is not None:
            self.conn.close()

        self.conn = None

    def __del__(self):
        self.close()

    def _check_open(self, open_and_close):
        if open_and_close:
            self.connect()
        else:
            if self.conn is None:
                raise Exception(
                    """Create connection first
                    (use self.connect() and other
                    functions needed for startup)
                    """
                )

    def _check_commit_and_close(self, commit=True, open_and_close=False):
        if commit:
            self.conn.commit()

        if open_and_close:
            self.close()

    def execute_statement(self, statement, commit=True, open_and_close=False):
        self._check_open(open_and_close)

        cursor = self.conn.cursor()
        cursor.execute(statement)

        self._check_commit_and_close(commit, open_and_close)
        return cursor

    def enable_vector_extension(self):
        self.execute_statement(
            "CREATE EXTENSION IF NOT EXISTS vector", open_and_close=True
        )

    def register_vector_in_conn(self):
        register_vector(self.conn)

    def insert_values(
        self,
        data_list,
        sql_columns,
        register_vector_conn=True,
        commit=True,
        open_and_close=False,
    ):
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
        """Obtiene el top 3 documentos similares a un query determinado
        por la distancia definida en el índice

        Parameters:
            embedder (Embedder): Objeto instanciado de la clase Embedder con,
            su configuración.
            user_input (str): String con la consulta del usuario
            schema (str): Schema de postgres en donde está la tabla
              con los embeddings
            table_name (str): Nombre de la tabla con los embeddings
            k (int): Número de documentos a recopilar de la base de datos
            register_vector_before (bool): indicador de si debe usarse
              register_vector (True) o no. Si no se registró al vector
              antes o se cerró la conexión entonces es
              necesario ponerloen True
            custom_where (str): Condición where para prefiltrar datos de
              la tabla de embeddings
            columns (str): Comma separated strings of columns to retrieve
            register_vector_conn (bool): indicador de si debe usarse
              register_vector (True) o no. Si no se registró al vector
              antes o se cerró la conexión entonces
              es necesario ponerlo en True
            commit (bool): indicador de si se debe hacer un commit en
              la operación al final o no
            open_and_close (bool): indicador de si debe auto gestionar
              el inicio y final de la conexión
            col_name_embedding (str): nombre de la columna que contiene los embeddings
            partition_by (str|None): columna de partición si se desea solo un resultado por
              ventana de partición. Enviar None de otro modo

        Returns:
            top_docs (List[Tuple]): Lista que contiene tantas tuplas
            como k. Cada uno indicando su distancia al final
            junto a varios atributos predefinidos
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

            args = (self.embedding_array, )
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
