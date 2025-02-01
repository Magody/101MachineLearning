import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import PyPDF2
import pandas as pd
import boto3
from langchain.embeddings import BedrockEmbeddings


class Embedder:
    def __init__(self, configuration, encoder):
        self.config_model = configuration["model"]
        self.config_splitter = configuration["splitter"]
        self.encoder = encoder
        self._text_splitter = None

    def get_splitter(self):
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config_splitter["chunk_size"],
                chunk_overlap=self.config_splitter["chunk_overlap"],
                length_function=self.config_splitter["length_function"],
                add_start_index=self.config_splitter["add_start_index"],
            )
        return self._text_splitter

    def split_and_explode(self, df_to_split_explode, column_name_content_ciiu):
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
        if check_cost:
            total_cost, total_tokens = Embedder.get_total_embeddings_cost_df(
                df_to_embed, column_name_content=column_name_content_ciiu
            )
            print(
                "Precio estimado DF = $"
                + str(total_cost)
                + f" para {total_tokens} tokens"
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
        documents_text_only = None

        if document_type.lower() == "pdf":
            """
            reader = PyPDF2.PdfReader(path_file)
            print("Pages PDF: ", len(reader.pages))
            pages_text_only = [
                reader.pages[i].extract_text() for i in range(len(reader.pages))
            ]

            # print(pages_text_only_full[0:100], len(pages_text_only_full))
            documents = self.get_splitter().create_documents(
                [" ".join(pages_text_only)], [{"filename": path_file}]
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
            """
            raise Exception("Not implemented")
        else:
            raise Exception("Not implemented")

        return documents_text_only

    def embed_pdf(self, path_file, check_cost=True, preprocess_function=None):
        if check_cost:
            chunks_pdf = self.split_document("PDF", path_file)

            total_tokens = Embedder.num_tokens_from_string("".join(chunks_pdf))
            total_cost = Embedder.get_embedding_cost(total_tokens)
            print(
                "Precio estimado PDF: $"
                + str(total_cost)
                + f" para {total_tokens} tokens"
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
        if not string:
            return 0
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def get_embedding_cost(num_tokens):
        return num_tokens / 1000 * 0.0001

    @staticmethod
    def get_total_embeddings_cost_df(df, column_name_content="content"):
        total_tokens = 0
        for i in df.index:
            text = df[column_name_content][i]
            token_len = Embedder.num_tokens_from_string(text)
            total_tokens = total_tokens + token_len
        total_cost = Embedder.get_embedding_cost(total_tokens)
        return total_cost, total_tokens


class EmbedderAWS(Embedder):
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
