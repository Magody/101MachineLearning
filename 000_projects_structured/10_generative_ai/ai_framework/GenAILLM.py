from enum import Enum

import boto3
import httpx
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat

from langchain_openai import ChatOpenAI, OpenAI
import pandas as pd
import json

from ai_framework.LoggingAndTelemetry import Logging, EnumLogs
from ai_framework.Embedder import Embedder


class EnumGenAIPlatforms(Enum):
    """Enumeration of available generative AI platforms"""
    PLATFORM_AWS_BEDROCK = "AWS-BEDROCK"
    PLATFORM_OPENAI = "OPENAI"
    PLATFORM_OTHER = "OTHER"


class EnumGenAIModelsIds(Enum):
    """Enumeration of generative AI model identifiers, as a parent class"""
    pass


class EnumGenAIModelsIdsBedrock(EnumGenAIModelsIds):
    """Enumeration of Bedrock generative AI model identifiers"""
    MODEL_CLAUDE_INSTANT = "anthropic.claude-instant-v1"
    MODEL_CLAUDE_V2 = "anthropic.claude-v2"
    MODEL_CLAUDE_V2_1 = "anthropic.claude-v2:1"
    MODEL_CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    MODEL_CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    MODEL_AMAZON_TITAN_EXPRESS = "amazon.titan-text-express-v1"
    MODEL_JURASSIC_2_MID = "ai21.j2-mid-v1"
    MODEL_JURASSIC_2_ULTRA = "ai21.j2-ultra-v1"
    MODEL_COHERE_COMMAND = "cohere.command-text-v14"
    MODEL_MISTRAL_AI_7B_INSTRUCT = "mistral.mistral-7b-instruct-v0:2"
    MODEL_MISTRAL_AI_8X7B_INSTRUCT = "mistral.mixtral-8x7b-instruct-v0:1"


class EnumGenAIModelsIdsOpenAI(EnumGenAIModelsIds):
    """Enumeration of OpenAI generative AI model identifiers"""
    MODEL_GPT_3_5_INSTRUCT = "gpt-3.5-turbo-instruct"
    MODEL_CHAT_GPT_3_5_TURBO = "gpt-3.5-turbo"
    MODEL_CHAT_GPT_3_5_TURBO_16k = "gpt-3.5-turbo-0125"

    MODEL_CHAT_GPT_4 = "gpt-4"
    MODEL_CHAT_GPT_4_32k = "gpt-4-32k"
    MODEL_CHAT_GPT_4_TURBO_128k = "gpt-4-1106-preview"
    MODEL_CHAT_GPT_4_TURBO_128k_VISION = "gpt-4-1106-vision-preview"


class EnumCostMode(Enum):
    """Enumeration of cost calculation modes"""
    COST_INPUT_TOKENS = "COST_INPUT_TOKENS"
    COST_OUTPUT_TOKENS = "COST_OUTPUT_TOKENS"


class GenAILLM:
    """
    Component responsible for storing and executing a large language model.

    Attributes:
        DELTA_EXTRA_TOKENS_FOR_COST (int): The additional number of tokens to consider for cost calculation.
        llm_model: The large language model instance.
        platform (EnumGenAIPlatforms): The platform on which the large language model is hosted.
        cost_per_1k_input (float): The cost per 1000 input tokens.
        cost_per_1k_output (float): The cost per 1000 output tokens.
        input_tokens (int): The number of input tokens.
        input_expected_cost (float): The expected cost of the input.
        output_tokens (int): The number of output tokens.
        output_expected_cost (float): The expected cost of the output.
        verbose_level (EnumLogs.LOG_LEVEL): The verbosity level for logging.
        model_id (str): The identifier of the large language model.
    """

    DELTA_EXTRA_TOKENS_FOR_COST = 2

    def __init__(
        self,
        platform: EnumGenAIPlatforms,
        model_id: EnumGenAIModelsIds,
        parameters_inference: dict = {
            "max_tokens": 512,
            "temperature": 0.2,
            "top_k": 40,
            "top_p": 0.2,
            "stop_sequences": ["\n\nHuman"],
        },
        platform_configuration: dict = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "region_name": "us-east-1",
            "OPENAI_API_KEY": "sk-",
        },
        verify_ssl: bool = True,
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        metadata: dict = {
            "precio_input": -1,
            "precio_output": -1,
        },  # Also can be pd.DataFrame
    ) -> None:
        """
        Initialize the GenAILLM component.

        Args:
            platform (EnumGenAIPlatforms): The platform for the generative AI model.
            model_id (EnumGenAIModelsIds): The identifier of the generative AI model.
            parameters_inference (dict, optional): The parameters for the inference process. Defaults to a dictionary with default values.
            platform_configuration (dict, optional): The configuration for the platform. Defaults to a dictionary with empty values.
            verify_ssl (bool, optional): Whether to verify the SSL certificate. Defaults to True.
            verbose_level (EnumLogs, optional): The verbosity level for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.
            metadata (dict, optional): The metadata for the model, including input and output prices. Defaults to a dictionary with -1 values.
        """

        if 'max_tokens' in parameters_inference and platform == EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK:
            parameters_inference['max_tokens_to_sample'] = parameters_inference['max_tokens']

        self.platform = platform
        self.model_id = model_id
        self.parameters_inference = parameters_inference
        self.platform_configuration = platform_configuration
        self.verify_ssl = verify_ssl
        self.verbose_level = verbose_level
        self.is_chat_model = model_id in [
            EnumGenAIModelsIdsOpenAI.MODEL_GPT_3_5_INSTRUCT,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO_16k,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4_32k,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4_TURBO_128k,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4_TURBO_128k_VISION,
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_HAIKU,
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_SONNET
        ]
        self.metadata = metadata

        if isinstance(self.metadata, pd.DataFrame):
            filter_mask = (self.metadata["plataforma"] == platform.value) & (
                self.metadata["modelo_id"] == model_id.value
            )
            self.metadata = json.loads(
                self.metadata[filter_mask].to_json(orient="records")
            )[0]

        # print(self.metadata)

        assert isinstance(self.metadata, dict) and len(self.metadata) >= 2

        self.cost_per_1k_input = float(self.metadata["precio_input"])
        self.cost_per_1k_output = float(self.metadata["precio_output"])

        assert self.cost_per_1k_input > 0 and self.cost_per_1k_output > 0

        self.verbose = self.verbose_level.value >= EnumLogs.LOG_LEVEL_ERROR.value
        self.build_model()
        Logging.log(
            f"Modelo {self.model_id} construido",
            self.verbose_level,
            EnumLogs.LOG_LEVEL_DEBUG,
        )

    def build_model(self):
        """Build the language model based on the specified platform and model identifier."""
        self.system_prefix = "System"

        if self.platform.value == EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK.value:

            if self.model_id.value == EnumGenAIModelsIdsBedrock.MODEL_AMAZON_TITAN_EXPRESS.value:
                inference_modifier = {
                    "temperature": self.parameters_inference["temperature"],
                    "maxTokenCount": self.parameters_inference["max_tokens_to_sample"],
                    "topP": self.parameters_inference["top_p"],
                    "stopSequences": self.parameters_inference["stop_sequences"],
                }
                self.ai_prefix = "Assistant"
                self.human_prefix = "User"

            elif self.model_id.value == EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2.value:

                inference_modifier = {
                    "temperature": self.parameters_inference["temperature"],
                    "max_tokens_to_sample": self.parameters_inference[
                        "max_tokens_to_sample"
                    ],
                    "top_p": self.parameters_inference["top_p"],
                    "top_k": self.parameters_inference["top_k"],
                    "stop_sequences": self.parameters_inference["stop_sequences"],
                }
                self.ai_prefix = "Assistant"
                self.human_prefix = "Human"

            elif self.model_id.value in (EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_HAIKU.value, EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_SONNET.value):

                inference_modifier = {
                    "temperature": self.parameters_inference["temperature"],
                    "max_tokens": self.parameters_inference[
                        "max_tokens_to_sample"
                    ],
                    "top_p": self.parameters_inference["top_p"],
                    "top_k": self.parameters_inference["top_k"],
                    "stop_sequences": self.parameters_inference["stop_sequences"],
                }
                self.ai_prefix = "Assistant"
                self.human_prefix = "Human"

            elif self.model_id.value in (EnumGenAIModelsIdsBedrock.MODEL_MISTRAL_AI_7B_INSTRUCT.value, EnumGenAIModelsIdsBedrock.MODEL_MISTRAL_AI_8X7B_INSTRUCT.value):

                inference_modifier = {
                    "temperature": self.parameters_inference["temperature"],
                    "max_tokens": self.parameters_inference[
                        "max_tokens_to_sample"
                    ],
                    "top_p": self.parameters_inference["top_p"],
                    "top_k": self.parameters_inference["top_k"],
                    # "stop_sequences": self.parameters_inference["stop_sequences"],
                }
                self.ai_prefix = "Assistant"
                self.human_prefix = "Human"

            else:
                inference_modifier = {
                    "temperature": self.parameters_inference["temperature"],
                    "max_tokens_to_sample": self.parameters_inference[
                        "max_tokens_to_sample"
                    ],
                    "top_p": self.parameters_inference["top_p"],
                    "top_k": self.parameters_inference["top_k"],
                    "stop_sequences": self.parameters_inference["stop_sequences"],
                }

            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.platform_configuration["region_name"],
                aws_access_key_id=self.platform_configuration["aws_access_key_id"],
                aws_secret_access_key=self.platform_configuration[
                    "aws_secret_access_key"
                ],
                verify=self.verify_ssl,
            )

            if self.model_id.value in (EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_HAIKU.value, EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_SONNET.value):

                self.llm_model = BedrockChat(
                    model_id=self.model_id.value,
                    client=self.client,
                    model_kwargs=inference_modifier,
                    verbose=self.verbose,
                )

            else:

                self.llm_model = Bedrock(
                    model_id=self.model_id.value,
                    client=self.client,
                    model_kwargs=inference_modifier,
                    verbose=self.verbose,
                )

        elif self.platform.value == EnumGenAIPlatforms.PLATFORM_OPENAI.value:

            model_kwargs = {
                key: value
                for key, value in self.parameters_inference.items()
                if key not in ["temperature", "max_tokens", "top_p"]
            }

            self.http_client = httpx.Client(verify=self.verify_ssl)

            if self.is_chat_model:
                self.llm_model = ChatOpenAI(
                    model=self.model_id.value,
                    temperature=self.parameters_inference["temperature"],
                    max_tokens=self.parameters_inference["max_tokens"],
                    model_kwargs=model_kwargs,
                    verbose=self.verbose,
                    http_client=self.http_client,
                )
            else:
                self.llm_model = OpenAI(
                    model=self.model_id.value,
                    temperature=self.parameters_inference["temperature"],
                    max_tokens=self.parameters_inference["max_tokens"],
                    model_kwargs=model_kwargs,
                    verbose=self.verbose,
                    http_client=self.http_client,
                )

        else:
            raise Exception(f"NOT IMPLEMENTED: {self.platform}")

    def calculate_tokens(self, text: str):
        """
        Calculate the number of tokens in the given text.

        Args:
            text (str): The input text.

        Returns:
            int: The number of tokens in the text.
        """
        if self.platform.value == EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK.value:
            return self.llm_model.get_num_tokens(text)

        elif self.platform.value == EnumGenAIPlatforms.PLATFORM_OPENAI.value:
            return Embedder.num_tokens_from_string(text)

    def calculate_cost(self, expected_tokens: int, mode_cost: EnumCostMode):
        """
        Calculate the cost based on the expected number of tokens and the cost mode.

        Args:
            expected_tokens (int): The expected number of tokens.
            mode_cost (EnumCostMode): The cost mode (input tokens or output tokens).

        Returns:
            float: The calculated cost.
        """
        if mode_cost.value == EnumCostMode.COST_INPUT_TOKENS.value:
            return expected_tokens / 1000 * self.cost_per_1k_input
        elif mode_cost.value == EnumCostMode.COST_OUTPUT_TOKENS.value:
            return expected_tokens / 1000 * self.cost_per_1k_output
        raise Exception(f"{mode_cost} not supported cost")

    def invoke(self, input, return_completion_only=True):
        """
        Invoke the large language model with the given input.

        Args:
            input (str): The input text.
            return_completion_only (bool, optional): Whether to return only the completion or the full result. Defaults to True.

        Returns:
            str: The result of the model invocation.
        """
        Logging.log(
            f"Invocando modelo {self.model_id} con {input}",
            self.verbose_level,
            EnumLogs.LOG_LEVEL_DEBUG,
        )
        result = self.llm_model.invoke(input)
        if return_completion_only:

            if hasattr(result, "content"):
                result = result.content.strip()
            else:
                result = str(result).strip()

        self.input_tokens = self.calculate_tokens(input)
        self.input_expected_cost = self.calculate_cost(
            self.input_tokens + GenAILLM.DELTA_EXTRA_TOKENS_FOR_COST,
            EnumCostMode.COST_INPUT_TOKENS,
        )

        self.output_tokens = self.calculate_tokens(result)
        self.output_expected_cost = self.calculate_cost(
            self.output_tokens + GenAILLM.DELTA_EXTRA_TOKENS_FOR_COST,
            EnumCostMode.COST_OUTPUT_TOKENS,
        )

        return result

    def generate_responses_from_dataframe(self, q_sentences):
        """
        Generate responses from a list of input sentences.

        Args:
            q_sentences (list): A list of input sentences.

        Returns:
            list: A list of generated responses.
        """
        responses = []

        # paralelizar
        for question in q_sentences:
            response = self.invoke(
                question, return_completion_only=True, save_in_memory=False
            )
            responses.append(response)
        return responses

    @staticmethod
    def build_predefined_instruct(
        platform_configuration,
        verbose_level=EnumLogs.LOG_LEVEL_DEBUG,
        platform=EnumGenAIPlatforms.PLATFORM_OPENAI,
        model_id=EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO,
    ):
        """
        Build a predefined instance of GenAILLM with the specified configuration.

        Args:
            platform_configuration (dict): The platform configuration.
            verbose_level (EnumLogs.LOG_LEVEL, optional): The verbosity level for logging. Defaults to EnumLogs.LOG_LEVEL_DEBUG.
            platform (EnumGenAIPlatforms, optional): The platform for the large language model. Defaults to EnumGenAIPlatforms.PLATFORM_OPENAI.
            model_id (EnumGenAIModelsIdsOpenAI, optional): The identifier of the large language model. Defaults to EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO.

        Returns:
            GenAILLM: The created instance of GenAILLM.
        """
        return GenAILLM(
            platform=platform,
            model_id=model_id,
            parameters_inference={
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 0.1,
            },
            platform_configuration=platform_configuration,
            verify_ssl=True,
            verbose_level=verbose_level,
        )
