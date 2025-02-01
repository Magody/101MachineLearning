from ai_framework.GenAILLM import (
    EnumGenAIModelsIdsBedrock,
    # EnumGenAIModelsIdsOpenAI,
    EnumGenAIPlatforms,
    GenAILLM,
)
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAIPrompt import GenAIPrompt
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging
import os
import sys
import pandas as pd

from ai_framework.test.utils import get_mode, get_file_in_framework

mode = get_mode(sys.argv)


def node_gen_ai_chain_execution(system_prompt):

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
            "top_k": 5,
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

    gen_ai_memory = GenAIMemory()

    def func_forward_and_save_chain(node: GenAINodeChain, shared_state, gen_ai_memory):
        Logging.log(f"RUNNING: {node.id}", node.verbose_level)
        chain_result = node.get_chain_result(shared_state, gen_ai_memory)
        return {
            **shared_state,
            "intermediate_steps_bp": [(node.id, "default")],
            **chain_result,
            "node_outcome": chain_result,
        }

    gen_ai_memory.reboot()

    node_direct_prompt = GenAINodeChain(
        "direct_prompt",
        gen_ai_llm,
        gen_ai_prompt=GenAIPrompt(
            """{system_prompt}
            """,
            partials={},
        ),
        gen_ai_output_parser=None,
        func_invoke=func_forward_and_save_chain,
        verbose_level=verbose_level,
    )

    node_result = node_direct_prompt.invoke(
        {"system_prompt": system_prompt}, gen_ai_memory
    )

    return node_direct_prompt, node_result[node_direct_prompt.output_key]


def test_node_gen_ai_chain_execution():
    if mode == "debug":
        assert len([1 for _ in range(4)]) == 4
    elif mode in ("test", "prod"):
        node_chain, output_text = node_gen_ai_chain_execution(
            "De qué color es el cielo"
        )
        assert len(output_text) > 0
        assert node_chain.total_input_tokens > 0
        assert node_chain.total_input_tokens_expected_cost > 0

        assert node_chain.total_output_tokens > 0
        assert node_chain.total_output_tokens_expected_cost > 0

        assert len(node_chain.absolute_node_input_history) > 0
        assert len(node_chain.status_history["INPUT_TOKENS"]) > 0
        assert len(node_chain.status_history["OUTPUT_TOKENS"]) > 0


if mode in ("test", "prod"):
    test_node_gen_ai_chain_execution()
