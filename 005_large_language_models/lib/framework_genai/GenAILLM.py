from enum import Enum

import boto3
from langchain_community.llms.bedrock import Bedrock
from langchain_openai import ChatOpenAI, OpenAI
from .LoggingAndTelemetry import Logging, EnumLogs
from langchain.chains import SequentialChain, LLMChain
from langchain.chains import ConversationChain
from .GenAIMemory import GenAIMemory, EnumMemoryType
from .GenAIPrompt import GenAIPrompt

class EnumGenAIChainType(Enum):
    CHAIN_TYPE_COMMON = "LLMChain"
    CHAIN_TYPE_CONVERSATIONAL = "ConversationalChain"

class EnumGenAIPlatforms(Enum):
    PLATFORM_AWS_BEDROCK = "AWS-BEDROCK"
    PLATFORM_OPENAI = "OPENAI"
    PLATFORM_OTHER = "OTHER"

class EnumGenAIModelsIds(Enum):
    pass

class EnumGenAIModelsIdsBedrock(EnumGenAIModelsIds):
    MODEL_CLAUDE_V2 = "anthropic.claude-v2"
    MODEL_AMAZON_TITAN_EXPRESS = "amazon.titan-text-express-v1"

class EnumGenAIModelsIdsOpenAI(EnumGenAIModelsIds):
    MODEL_CHAT_GPT_3_5_TURBO = "gpt-3.5-turbo"
    MODEL_CHAT_GPT_3_5_TURBO_16k = "gpt-3.5-turbo-0125"
    MODEL_CHAT_GPT_4 = "gpt-4"
    MODEL_GPT_3_5_INSTRUCT = "gpt-3.5-turbo-instruct"

class GenAILLM:

    def __init__(
        self, 
        platform:EnumGenAIPlatforms,
        model_id:EnumGenAIModelsIds,
        parameters_inference:dict = {
            'max_tokens_to_sample':512, 
            "temperature":0.5,
            "top_k":250,
            "top_p":1,
            "stop_sequences": ["\n\nHuman"]
        },
        ai_prefix = None,
        human_prefix = None,
        component_memory:GenAIMemory = None,
        platform_configuration:dict = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "region_name": "us-east-1",
            "OPENAI_API_KEY": "sk-"
        },
        verify_ssl:bool = True,
        verbose_level:EnumLogs = EnumLogs.LOG_LEVEL_INFO
    ) -> None:
        
        self.platform = platform
        self.model_id = model_id
        self.parameters_inference = parameters_inference
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.component_memory = component_memory
        self.platform_configuration = platform_configuration
        self.verify_ssl = verify_ssl
        self.verbose_level = verbose_level
        self.is_chat_model = model_id in [
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO_16k
        ]

        if self.component_memory is None:
            if self.is_chat_model:
                self.component_memory = GenAIMemory(EnumMemoryType.MEMORY_CHAT_BUFFER, ai_prefix, human_prefix)
            else:
                self.component_memory = GenAIMemory(EnumMemoryType.MEMORY_SIMPLE_MEMORY, ai_prefix, human_prefix)

        self.verbose = (self.verbose_level.value >= EnumLogs.LOG_LEVEL_ERROR.value)
        self.build_model()
        Logging.log(f"Modelo {self.model_id} construido", self.verbose_level, EnumLogs.LOG_LEVEL_DEBUG)

    def build_model(self):
        
        if self.platform == EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK:
        
            if self.model_id == EnumGenAIModelsIdsBedrock.MODEL_AMAZON_TITAN_EXPRESS:
                inference_modifier = {
                    'temperature': self.parameters_inference["temperature"], 
                    "maxTokenCount": self.parameters_inference["max_tokens_to_sample"], 
                    "topP": self.parameters_inference["top_p"], 
                    "stopSequences": self.parameters_inference["stop_sequences"], 
                }
                self.ai_prefix = "Assistant"
                self.human_prefix = "User"

            elif self.model_id == EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2:
                
                inference_modifier = {
                    'temperature': self.parameters_inference["temperature"],
                    "max_tokens_to_sample": self.parameters_inference["max_tokens_to_sample"],
                    "top_p": self.parameters_inference["top_p"],
                    "top_k": self.parameters_inference["top_k"],
                    "stop_sequences": self.parameters_inference["stop_sequences"]
                }
                self.ai_prefix = "Assistant"
                self.human_prefix = "Human"

            else:
                inference_modifier = {
                    "temperature": self.parameters_inference["temperature"],
                    'max_tokens_to_sample': self.parameters_inference["max_tokens_to_sample"],
                    "top_p": self.parameters_inference["top_p"],
                    "top_k": self.parameters_inference["top_k"],
                    "stop_sequences": self.parameters_inference["stop_sequences"]
                }
            

            client_bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.platform_configuration["aws_region"],
                aws_access_key_id=self.platform_configuration["aws_access_key"],
                aws_secret_access_key=self.platform_configuration["aws_secret_key"],
                verify=self.verify_ssl
            )

            self.llm_model = Bedrock(
                model_id=self.model_id.value, 
                client=client_bedrock_runtime, 
                model_kwargs=inference_modifier,
                verbose=self.verbose
            )

        elif self.platform == EnumGenAIPlatforms.PLATFORM_OPENAI:

            model_kwargs = {key: value for key, value in self.parameters_inference.items() if key not in ["temperature", "max_tokens", "top_p"]}

            if self.is_chat_model:
                self.llm_model = ChatOpenAI(
                    model=self.model_id.value,
                    temperature=self.parameters_inference["temperature"],
                    max_tokens=self.parameters_inference["max_tokens"],
                    model_kwargs=model_kwargs,
                    verbose=self.verbose
                )
            else:
                self.llm_model = OpenAI(
                    model=self.model_id.value,
                    temperature=self.parameters_inference["temperature"],
                    max_tokens=self.parameters_inference["max_tokens"],
                    model_kwargs=model_kwargs,
                    verbose=self.verbose
                )
            
        else:
            raise Exception(f"NOT IMPLEMENTED: {self.platform}")
        
    def invoke(self, input, return_completion_only=True, save_in_memory=False):
        Logging.log(f"Invocando modelo {self.model_id} con {input}", self.verbose_level, EnumLogs.LOG_LEVEL_DEBUG)
        result = None
        if return_completion_only:

            if self.platform == EnumGenAIPlatforms.PLATFORM_OPENAI:
                result = self.llm_model.invoke(input).content.strip()
                
            else:
                result = self.llm_model.invoke(input).strip()
        else:  
            result = self.llm_model.invoke(input)

        if save_in_memory:
            self.component_memory.add_registry(input, self.human_prefix)
            self.component_memory.add_registry(result, self.ai_prefix)
        
        return result
    
    def create_chain(self, gen_ai_prompt:GenAIPrompt, chain_type:EnumGenAIChainType=EnumGenAIChainType.CHAIN_TYPE_COMMON, output_key="output"):
        
        if chain_type == EnumGenAIChainType.CHAIN_TYPE_COMMON:
            return LLMChain(
                llm=self.llm_model, 
                prompt=gen_ai_prompt.prompt_template,
                output_key=output_key,
                verbose=self.verbose
            )
        elif chain_type == EnumGenAIChainType.CHAIN_TYPE_CONVERSATIONAL:
            return ConversationChain(
                llm=self.llm_model,
                prompt=gen_ai_prompt.prompt_template,
                memory=self.component_memory.memory,
                verbose=self.verbose
            )
        else:
            raise Exception("NOT IMPLEMENTED")
    
    def create_sequential_chain(self, chains, input_variables, partial_variables, output_variables=["final_output"]):
        return SequentialChain(
            memory=self.component_memory.memory,
            chains=chains,
            input_variables=input_variables,
            output_variables=output_variables,
            partial_variables=partial_variables,
            verbose=self.verbose
        )
    
    @staticmethod
    def build_predefined_instruct(platform_configuration, verbose_level=EnumLogs.LOG_LEVEL_DEBUG, platform=EnumGenAIPlatforms.PLATFORM_OPENAI, model_id=EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO):
        ai_prefix = "Human"
        human_prefix = "Assistant"

        return GenAILLM(
            platform = platform,
            model_id = model_id,
            parameters_inference = {
                'max_tokens': 1024, 
                "temperature": 0,
                "top_p": 0.1
            },
            ai_prefix = ai_prefix,
            human_prefix = human_prefix,
            component_memory=GenAIMemory(EnumMemoryType.MEMORY_CHAT_BUFFER, ai_prefix, human_prefix),
            platform_configuration = platform_configuration,
            verify_ssl=True,
            verbose_level=verbose_level
        )
    
