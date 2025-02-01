from enum import Enum
import pandas as pd

from ai_framework.Embedder import Embedder
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINodeChain import GenAINodeChain, EnumGenAINodeChainType
from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAIOutputParser import GenAIOutputParser
from ai_framework.GenAIPrompt import GenAIPrompt
from ai_framework.LoggingAndTelemetry import EnumLogs
from ai_framework.VectorDBHandler import VectorDBHandler


class EnumRagStrategy(Enum):
    RAG_COMMON = "Common"
    RAG_HyDE = "HyDE"


class GenAINodeChainRAG(GenAINodeChain):
    """
    A subclass of GenAINodeChain that implements Retrieval Augmented Generation (RAG).

    This class is responsible for managing the retrieval of relevant documents from a vector database
    and incorporating them into the input for the language model.

    Args:
        id (str): The unique identifier for this node chain.
        gen_ai_llm (GenAILLM): The language model to be used.
        gen_ai_prompt (GenAIPrompt): The prompt to be used for the language model.
        gen_ai_output_parser (GenAIOutputParser, optional): The output parser to be used.
        func_invoke (callable, optional): A custom function to be invoked during the chain execution.
        verbose_level (EnumLogs, optional): The logging level for this node chain.
        inject_memory_common (bool, optional): Whether to inject common memory into the input.
        inject_memory_chat (bool, optional): Whether to inject chat history into the input.
        vars_autofill (dict, optional): A dictionary of variable names to be automatically filled in the input.
        output_key (str, optional): The key for the output of the chain.
        k (int, optional): The number of similar documents to retrieve.
        rag_strategy (EnumRagStrategy, optional): The RAG strategy to be used.
        embedder (Embedder): The embedder to be used for document retrieval.
        vector_db (VectorDBHandler): The vector database handler to be used for document retrieval.
    """

    def __init__(
        self,
        id: str,
        gen_ai_llm: GenAILLM,
        gen_ai_prompt: GenAIPrompt,
        gen_ai_output_parser: GenAIOutputParser = None,
        func_invoke=None,
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        inject_memory_common=False,
        inject_memory_chat=False,
        vars_autofill: dict = {
            "context": "context",
            "chat_history": "chat_history",
            "format_instructions": "format_instructions",
        },
        output_key="auto",
        k: int = 7,
        rag_strategy: EnumRagStrategy = EnumRagStrategy.RAG_COMMON,
        embedder: Embedder = None,
        vector_db: VectorDBHandler = None,
    ):
        super().__init__(
            id,
            gen_ai_llm,
            gen_ai_prompt,
            gen_ai_output_parser,
            func_invoke,
            EnumGenAINodeChainType.CHAIN_TYPE_COMMON_PLUS_RAG,
            verbose_level=verbose_level,
            inject_memory_common=inject_memory_common,
            inject_memory_chat=inject_memory_chat,
            vars_autofill=vars_autofill,
            output_key=output_key,
        )
        self.k = k
        self.rag_strategy = rag_strategy
        assert embedder is not None and vector_db is not None
        self.embedder = embedder
        self.vector_db = vector_db

    def retrieval(self, query, schema, table_name, custom_where, columns, col_name_embedding, partition_by):
        """
        Retrieves similar documents from the vector database based on the given query.

        Args:
            query (str): The user's input query.
            schema (str): The schema name of the table in the vector database.
            table_name (str): The name of the table in the vector database.
            custom_where (str): A custom WHERE clause for the database query.
            columns (list): The list of columns to retrieve from the database.
            col_name_embedding (str): The name of the column containing the document embeddings.
            partition_by (str, optional): The column to partition the results by.

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved documents and their similarity scores.
        """

        if "scoreSim" not in columns:
            columns.append("scoreSim")

        columns_query = (
            ",".join(columns).replace(",scoreSim", "").replace("scoreSim,", "")
        )

        return pd.DataFrame(
            self.vector_db.get_similar_docs(
                self.embedder,
                user_input=query,
                schema=schema,
                table_name=table_name,
                k=self.k,
                custom_where=custom_where,
                columns=columns_query,
                register_vector_conn=True,
                commit=True,
                open_and_close=True,
                col_name_embedding=col_name_embedding,
                partition_by=partition_by,
            ),
            columns=columns,
        )

    def get_chain_result(
        self,
        input: dict,
        gen_ai_memory: GenAIMemory,
        schema_table: str,
        query_key="user_input",
        columns=[
            "contenido_original",
            "metadata_pregunta",
            "metadata_respuesta",
            "vectorizacion_algoritmo",
            "version",
            "scoreSim",
        ],
        custom_where="WHERE version = '1'",
        col_name_embedding="contenido_vector",
        col_name_original="contenido_original",
        partition_by=None,
    ):
        """
        Retrieves similar documents from the vector database and incorporates them into the input for the language model.

        Args:
            input (dict): The input dictionary for the chain.
            gen_ai_memory (GenAIMemory): The memory object to be used for the chain.
            schema_table (str): The schema and table name in the format "schema.table".
            query_key (str, optional): The key for the user's input query in the input dictionary.
            columns (list, optional): The list of columns to retrieve from the database.
            custom_where (str, optional): A custom WHERE clause for the database query.
            col_name_embedding (str, optional): The name of the column containing the document embeddings.
            col_name_original (str, optional): The name of the column containing the original document content.
            partition_by (str, optional): The column to partition the results by.

        Returns:
            dict: The result of the chain execution.
        """
        if "node_outcome" in input:
            input = {**input, **input["node_outcome"]}

        query = input[query_key]

        schema_table_split = schema_table.split(".")

        similar_docs_for_rag = self.retrieval(
            query,
            schema_table_split[0],
            schema_table_split[1],
            custom_where,
            columns,
            col_name_embedding,
            partition_by,
        )

        rag = "<documentos>"
        for i, (_, row) in enumerate(similar_docs_for_rag.iterrows()):
            doc = row[col_name_original]
            rag += f"<documento id='{i}'>{doc}</documento>\n"
        rag += "</documentos>"

        # similar_docs_for_rag.sort_values("scoreSim",ascending=False).head(10)
        input["rag"] = rag

        chain_result = self.chain.invoke(input)

        return self.pos_process_chain_result(chain_result, gen_ai_memory)
