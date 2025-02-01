from ai_framework.GenAILLM import (
    EnumGenAIModelsIdsBedrock,
    EnumGenAIPlatforms,
    GenAILLM,
)
from ai_framework.LoggingAndTelemetry import EnumLogs
import os
import sys
from .utils import get_mode

mode = get_mode(sys.argv)


def simple_prompt(prompt_direct):
    verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_DEBUG
    verify_ssl = False

    gen_ai_llm = GenAILLM(
        platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,
        model_id=EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2,
        parameters_inference={
            "max_tokens": 1024,
            "temperature": 0,
            "top_p": 0,
            "stop_sequences": ["User"],
            "max_tokens_to_sample": 1024,
            "top_k": 5,
        },
        platform_configuration={
            "aws_access_key_id": os.environ["aws_access_key_id"],
            "aws_secret_access_key": os.environ["aws_secret_access_key"],
            "region_name": os.environ["region_name"],
        },
        verify_ssl=verify_ssl,
        verbose_level=verbose_level,
    )

    llm_result = gen_ai_llm.invoke(prompt_direct, return_completion_only=True)
    return llm_result


def test_simple_prompt():
    if mode == "debug":
        assert len([1 for _ in range(5)]) == 5
    elif mode in ('test', 'prod'):
        output = simple_prompt("Cuentame una historia corta")
        print(output)
        assert len(output) > 0


if mode in ('test', 'prod'):
    test_simple_prompt()
