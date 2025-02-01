import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import boto3
from langchain.embeddings import BedrockEmbeddings
import PyPDF2


class Embedder:
    """
    The Embedder class is responsible for splitting and embedding text data.

    Attributes:
        config_model (dict): Configuration for the model.
        config_splitter (dict): Configuration for the text splitter.
        encoder (object): The encoder object used for embedding text.
        _text_splitter (object): The text splitter object.

    Methods:
        get_splitter(): Returns the text splitter object.
        split_and_explode(df_to_split_explode, column_name_content_ciiu): Splits and explodes the given DataFrame.
        embed_dataframe(df_to_embed, column_name_content_ciiu, check_cost=True): Embeds the text in the given DataFrame.
        split_document(document_type, path_file, preprocess_function=None): Splits the document based on the document type.
        embed_pdf(path_file, check_cost=True, preprocess_function=None): Embeds the text in a PDF document.
        num_tokens_from_string(string: str, encoding_name="cl100k_base"): Calculates the number of tokens in a given string.
        helper_generate_vectors_df(self, df, column_name_content, check_cost=True): Generates the embedding vectors for a DataFrame.
        get_embedding_cost(num_tokens, price_per_1k=0.0001): Calculates the cost of embedding a given number of tokens.
        get_total_embeddings_cost_df(df, column_name_content="content"): Calculates the total cost of embedding a DataFrame.
    """

    def __init__(self, configuration, encoder):
        self.config_model = configuration["model"]
        self.config_splitter = configuration["splitter"]
        self.encoder = encoder
        self._text_splitter = None

    def get_splitter(self):
        """
        Returns the text splitter object.

        If the text splitter object is not yet initialized, it creates a new one based on the configuration.

        Returns:
            object: The text splitter object.
        """
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config_splitter["chunk_size"],
                chunk_overlap=self.config_splitter["chunk_overlap"],
                length_function=self.config_splitter["length_function"],
                add_start_index=self.config_splitter["add_start_index"],
            )
        return self._text_splitter

    def split_and_explode(self, df_to_split_explode, column_name_content_ciiu):
        """
        Splits and explodes the given DataFrame.

        Args:
            df_to_split_explode (pandas.DataFrame): The DataFrame to be split and exploded.
            column_name_content_ciiu (str): The name of the column containing the content to be split.

        Returns:
            pandas.DataFrame: The split and exploded DataFrame.
        """
        # TODO: optimizar copia que se haga en el mismo dataframe
        column_name_content_ciiu_split = column_name_content_ciiu + "_split"
        # TODO: Paralelizable
        df_to_split_explode[column_name_content_ciiu_split] = df_to_split_explode[
            column_name_content_ciiu
        ].apply(self.get_splitter().split_text)
        # TODO: Paralelizable
        return (
            df_to_split_explode.drop([column_name_content_ciiu], axis=1)
            .explode(column_name_content_ciiu_split)
            .rename(columns={column_name_content_ciiu_split: column_name_content_ciiu})
        )

    def embed_dataframe(self, df_to_embed, column_name_content_ciiu, check_cost=True):
        """
        Embeds the text in the given DataFrame.

        Args:
            df_to_embed (pandas.DataFrame): The DataFrame to be embedded.
            column_name_content_ciiu (str): The name of the column containing the content to be embedded.
            check_cost (bool, optional): Whether to check the cost of embedding. Defaults to True.

        Returns:
            pandas.DataFrame: The DataFrame with the embedded text.
        """
        if check_cost:
            total_cost, total_tokens = Embedder.get_total_embeddings_cost_df(
                df_to_embed, column_name_content=column_name_content_ciiu
            )
            print(
                "Precio estimado DF = $" + str(total_cost) + f" para {total_tokens} tokens"
            )

        documents_embeddings = self.encoder.embed_documents(
            list(df_to_embed[column_name_content_ciiu].values)
        )

        return pd.concat(
            [
                df_to_embed.reset_index(),
                pd.Series(documents_embeddings)
                .to_frame()
                .rename(columns={0: "embeddings"}),
            ],
            axis=1,
        )

    def split_document(self, document_type, path_file, preprocess_function=None):
        """
        Splits the document based on the document type.

        Args:
            document_type (str): The type of the document (e.g., "pdf").
            path_file (str or dict): The path to the document or a dictionary containing the data reader and path.
            preprocess_function (callable, optional): A function to preprocess the document text. Defaults to None.

        Returns:
            list: The list of preprocessed document chunks.
        """
        documents_text_only = None

        if document_type.lower() == "pdf":
            path_filename = ""
            if isinstance(path_file, str):
                reader = PyPDF2.PdfReader(path_file)
                path_filename = path_file
            elif isinstance(path_file, dict):
                data_reader = path_file["data_reader"]
                path_filename = path_file["path_filename"]
                reader = PyPDF2.PdfReader(data_reader)

            print("Pages PDF: ", len(reader.pages))
            pages_text_only = [
                reader.pages[i].extract_text() for i in range(len(reader.pages))
            ]

            # print(pages_text_only_full[0:100], len(pages_text_only_full))
            documents = self.get_splitter().create_documents(
                [" ".join(pages_text_only)], [{"filename": path_filename}]
            )

            if preprocess_function is not None:
                documents_text_only = [
                    preprocess_function(documents[i].page_content)
                    for i in range(len(documents))
                ]
            else:
                documents_text_only = [
                    documents[i].page_content for i in range(len(documents))
                ]
        else:
            raise Exception("Not implemented")

        return documents_text_only

    def embed_pdf(self, path_file, check_cost=True, preprocess_function=None):
        """
        Embeds the text in a PDF document.

        Args:
            path_file (str or dict): The path to the PDF file or a dictionary containing the data reader and path.
            check_cost (bool, optional): Whether to check the cost of embedding. Defaults to True.
            preprocess_function (callable, optional): A function to preprocess the document text. Defaults to None.

        Returns:
            pandas.DataFrame: The DataFrame containing the embedded text.
        """
        if check_cost:
            chunks_pdf = self.split_document("PDF", path_file)

            total_tokens = Embedder.num_tokens_from_string("".join(chunks_pdf))
            total_cost = Embedder.get_embedding_cost(total_tokens)
            print(
                "Precio estimado PDF: $" + str(total_cost) + f" para {total_tokens} tokens"
            )

        documents_text_only = self.split_document("pdf", path_file, preprocess_function)
        documents_embeddings = self.encoder.embed_documents(documents_text_only)

        return pd.concat(
            [
                pd.Series(documents_text_only)
                .to_frame()
                .rename(columns={0: "content"}),
                pd.Series(documents_embeddings)
                .to_frame()
                .rename(columns={0: "embeddings"}),
            ],
            axis=1,
        )

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
        """
        Calculates the number of tokens in a given string.

        This uses the OpenAI count. Depending on the model, GenAILLM calculates its own token count and cost.
        Use this function with care.

        Args:
            string (str): The input string.
            encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base".

        Returns:
            int: The number of tokens in the input string.
        """
        if not string:
            return 0

        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception as error_encoding:
            print(f"Error {error_encoding} not supported")
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = len(encoding.encode(string))
        return num_tokens

    def helper_generate_vectors_df(self, df, column_name_content, check_cost=True):
        """
        Generates the embedding vectors for a DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            column_name_content (str): The name of the column containing the content to be embedded.
            check_cost (bool, optional): Whether to check the cost of embedding. Defaults to True.

        Returns:
            pandas.DataFrame: The DataFrame with the embedded text.
        """
        df = self.split_and_explode(
            df, column_name_content_ciiu=column_name_content
        ).reset_index()

        df = self.embed_dataframe(df, column_name_content, check_cost=check_cost)
        return df

    @staticmethod
    def get_embedding_cost(num_tokens, price_per_1k=0.0001):
        """
        Calculates the cost of embedding a given number of tokens.

        Args:
            num_tokens (int): The number of tokens to be embedded.
            price_per_1k (float, optional): The price per 1000 tokens. Defaults to 0.0001.

        Returns:
            float: The cost of embedding the given number of tokens.
        """
        return num_tokens / 1000 * price_per_1k

    @staticmethod
    def get_total_embeddings_cost_df(df, column_name_content="content"):
        """
        Calculates the total cost of embedding a DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            column_name_content (str, optional): The name of the column containing the content to be embedded. Defaults to "content".

        Returns:
            tuple: The total cost and the total number of tokens.
        """
        total_tokens = 0
        for i in df.index:
            text = df[column_name_content][i]
            token_len = Embedder.num_tokens_from_string(text)
            total_tokens = total_tokens + token_len
        total_cost = Embedder.get_embedding_cost(total_tokens)
        return total_cost, total_tokens


class EmbedderAWS(Embedder):
    """
    The EmbedderAWS class is a subclass of the Embedder class, which is responsible for embedding text data using AWS Bedrock.

    Attributes:
        client_bedrock (boto3.client): The AWS Bedrock client.

    Methods:
        __init__(self, configuration, encoder=None, verify=True): Initializes the EmbedderAWS object.
    """
    
    def __init__(self, configuration, encoder=None, verify=True):
        super().__init__(configuration, encoder)

        self.client_bedrock = boto3.client(
            "bedrock-runtime",
            region_name=configuration["provider"]["region"],
            aws_access_key_id=configuration["provider"]["credentials"][
                "aws_access_key_id"
            ],
            aws_secret_access_key=configuration["provider"]["credentials"][
                "aws_secret_access_key"
            ],
            verify=verify,
        )

        if encoder is None:
            self.encoder = BedrockEmbeddings(
                model_id=self.config_model["id"], client=self.client_bedrock
            )
