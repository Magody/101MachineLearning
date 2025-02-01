from ai_framework.GenAILLM import (
    EnumGenAIModelsIdsBedrock,
    EnumGenAIPlatforms,
    GenAILLM,
)
from ai_framework.LoggingAndTelemetry import EnumLogs
import os
import sys
import pandas as pd
from ai_framework.test.utils import get_mode, get_file_in_framework

mode = get_mode(sys.argv)


def simple_prompt(prompt_direct):
    verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_DEBUG
    verify_ssl = False

    gen_ai_llm = GenAILLM(
        platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,
        model_id=EnumGenAIModelsIdsBedrock.MODEL_AMAZON_TITAN_EXPRESS,
        parameters_inference={
            "max_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.1,
            "stop_sequences": ["User:"],
            "max_tokens_to_sample": 128,
            "top_k": 40,
        },
        platform_configuration={
            # "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "aws_access_key_id": os.environ["aws_access_key_id"],
            "aws_secret_access_key": os.environ["aws_secret_access_key"],
            "region_name": os.environ["region_name"],
        },
        verify_ssl=verify_ssl,
        verbose_level=verbose_level,
        metadata=pd.read_csv(get_file_in_framework("test/model_prices.csv"), delimiter=";"),
    )

    llm_result = gen_ai_llm.invoke(prompt_direct, return_completion_only=True)
    return llm_result, gen_ai_llm


def test_simple_prompt():
    if mode == "debug":
        assert len([1 for _ in range(5)]) == 5
    elif mode in ('test', 'prod'):
        output, gen_ai_llm = simple_prompt("Cuentame una historia corta")
        print(output)
        assert len(output) > 0
        assert gen_ai_llm.input_expected_cost > 0 and gen_ai_llm.output_expected_cost > 0


if mode in ('test', 'prod'):
    test_simple_prompt()
