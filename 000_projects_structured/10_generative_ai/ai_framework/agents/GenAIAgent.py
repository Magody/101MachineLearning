from typing import List
from enum import Enum
from ai_framework.GenAIPrompt import EnumPromptsBase, GenAIPrompt
from ai_framework.GenAILLM import (
    EnumGenAIModelsIds,
    GenAILLM,
    EnumGenAIPlatforms,
    EnumGenAIModelsIdsBedrock,
    EnumGenAIModelsIdsOpenAI,
)
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.agents.GenAIAgentTool import GenAIAgentTool
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAINode import GenAINodeCustom
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging
import pandas as pd

import os
import warnings


class EnumGenAIAgentType(Enum):
    AGENT_TYPE_REACT = "REACT"
    # TODO: decision graph agent


class EnumGenAIChatHistoryInjection(Enum):
    INJECTION_MODE_NOTHING = 0
    INJECTION_MODE_FULL = 1
    INJECTION_MODE_RELATIVE_TO_NAME = 2


class GenAIAgent(GenAINodeCustom):
    """
    The GenAIAgent class is a custom node in the AI framework that represents an agent capable of performing various tasks.

    Args:
        platform (EnumGenAIPlatforms): The platform the agent will use for its language model.
        model_id (EnumGenAIModelsIds): The ID of the language model the agent will use.
        tools (List[GenAIAgentTool]): A list of tools the agent can use to perform tasks.
        name (str): The name of the agent.
        maximum_iterations (int, optional): The maximum number of iterations the agent will perform. Defaults to None, which sets the maximum to the length of the tools list plus 2.
        func_error_handling (callable, optional): A function to handle errors that occur during the agent's execution.
        agent_type (EnumGenAIAgentType, optional): The type of agent. Defaults to EnumGenAIAgentType.AGENT_TYPE_REACT.
        chat_history_injection (EnumGenAIChatHistoryInjection, optional): The mode for injecting chat history into the agent's input. Defaults to EnumGenAIChatHistoryInjection.INJECTION_MODE_NOTHING.
        extra_system_prompt (str, optional): Additional system prompt for the agent.
        extra_parameters_inference (dict, optional): Additional parameters for the language model inference.
        verify_ssl (bool, optional): Whether to verify the SSL certificate. Defaults to False.
        verbose_level (EnumLogs, optional): The level of verbosity for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.
        vars_autofill (dict, optional): A dictionary of variables to automatically fill in the agent's input. Defaults to {"chat_history": "chat_history"}.
    """

    def __init__(
        self,
        platform: EnumGenAIPlatforms,
        model_id: EnumGenAIModelsIds,
        tools: List[GenAIAgentTool],
        name: str,
        maximum_iterations: int = None,
        func_error_handling=None,
        agent_type: EnumGenAIAgentType = EnumGenAIAgentType.AGENT_TYPE_REACT,
        chat_history_injection: EnumGenAIChatHistoryInjection = EnumGenAIChatHistoryInjection.INJECTION_MODE_NOTHING,
        extra_system_prompt: str = "",
        extra_parameters_inference: dict = {},
        verify_ssl: bool = False,
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        vars_autofill: dict = {"chat_history": "chat_history"},
    ) -> None:

        self.platform = platform
        self.model_id = model_id

        if self.model_id.value not in (
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_INSTANT.value,
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2.value,
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2_1.value,
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_HAIKU.value,
            EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_SONNET.value,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO.value,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO_16k.value,
            EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4.value,
        ):
            raise NotImplementedError(
                f"The model {self.model_id} is not supported by this library yet"
            )

        self.tools = tools

        self.tools_map = dict()
        for tool in tools:
            if isinstance(tool, GenAIAgentTool):
                self.tools_map[GenAIAgentTool._normalize_tool_name(tool.name)] = (
                    tool.func
                )

        self.name = name

        self.maximum_iterations = maximum_iterations
        if self.maximum_iterations is None:
            self.maximum_iterations = len(self.tools) + 2

        self.state = dict()
        self.func_error_handling = func_error_handling
        self.agent_type = agent_type

        self.vars_autofill = vars_autofill

        self.chat_history_injection = chat_history_injection
        self.extra_system_prompt = extra_system_prompt

        self.verbose_level = verbose_level
        self.verbose = self.verbose_level.value >= EnumLogs.LOG_LEVEL_DEBUG.value

        super().__init__(self.name, self.invoke, self.verbose_level)

        if agent_type == EnumGenAIAgentType.AGENT_TYPE_REACT:

            platform_configuration = dict()

            if self.platform.value == EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK.value:
                platform_configuration = {
                    "aws_access_key_id": os.environ["aws_access_key_id"],
                    "aws_secret_access_key": os.environ["aws_secret_access_key"],
                    "region_name": os.environ["region_name"],
                }

                # TODO: Check if it is a model by anthropic PROMPT_TEMPLATE_AGENT_REACT_TARGET_ANTHROPIC
                # Format of definitions in XML when ANTHROPIC model
                self.gen_ai_prompt_agent_react = GenAIPrompt(
                    EnumPromptsBase.PROMPT_TEMPLATE_AGENT_REACT_BY_CHAT_OPTIONS_ES.value,
                    partials={
                        "system_prompt": self.extra_system_prompt,
                        "tools_definition": GenAIAgentTool.get_tools_definitions(
                            self.tools
                        ),
                        "tools_names": GenAIAgentTool.get_tools_names(self.tools),
                        "action_human": GenAIAgentTool.get_tool_human().name,
                    },
                )

            elif self.platform.value == EnumGenAIPlatforms.PLATFORM_OPENAI.value:
                platform_configuration = {
                    "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                }

                self.gen_ai_prompt_agent_react = GenAIPrompt(
                    EnumPromptsBase.PROMPT_AGENT_REACT_CHAT.value,
                    partials={
                        "system_prompt": self.extra_system_prompt,
                        "tools": GenAIAgentTool.get_tools_definitions(tools),
                        "tools_names": GenAIAgentTool.get_tools_names(tools),
                        "action_human": GenAIAgentTool.get_tool_human().name,
                    },
                )

            path_metadata_prices = f"{os.getcwd()}/ai_framework/test/model_prices.csv"
            path_metadata_prices_lit_csv = f"{os.getcwd()}/model_prices.csv"

            if not os.path.isfile(path_metadata_prices):
                if os.path.isfile(path_metadata_prices_lit_csv):
                    path_metadata_prices = path_metadata_prices_lit_csv
                else:
                    raise Exception(
                        f"Path {path_metadata_prices_lit_csv} not found! Add the CSV in the project level {path_metadata_prices_lit_csv}"
                    )

            self.gen_ai_llm = GenAILLM(
                platform=self.platform,
                model_id=self.model_id,
                parameters_inference={
                    "max_tokens": 1024,
                    "temperature": 0,
                    "top_p": 0.1,
                    "stop_sequences": ["Observación:"],
                    "top_k": 50,
                    **extra_parameters_inference,
                },
                platform_configuration=platform_configuration,
                verify_ssl=verify_ssl,
                verbose_level=verbose_level,
                metadata=pd.read_csv(path_metadata_prices, delimiter=";"),
            )

            self.gen_ai_llm.verbose_level = EnumLogs.LOG_LEVEL_NOTHING
            self.gen_ai_llm.verbose = False

            self.node_agent_feed_forward = GenAINodeChain(
                self.name,
                self.gen_ai_llm,
                gen_ai_prompt=self.gen_ai_prompt_agent_react,
                gen_ai_output_parser=None,
                verbose_level=self.verbose_level,
            )

        else:
            raise Exception("NOT IMPLEMENTED")

    def find_tool(self, tool_name):
        """
        Finds the appropriate tool for the given tool name.

        Args:
            tool_name (str or list): The name of the tool to find.

        Returns:
            tuple: A tuple containing the normalized tool name and the corresponding tool function.
        """
        if isinstance(tool_name, list):
            if len(tool_name) == 0:
                # Error, giving last info. TODO: capture error outside and control outside
                tool_name = GenAIAgentTool.get_tool_human().name
            else:
                tool_name = tool_name[-1]

        tool_name = GenAIAgentTool._normalize_tool_name(tool_name)

        return tool_name, self.tools_map.get(tool_name, None)

    def invoke(self, node, input: dict, gen_ai_memory: GenAIMemory):
        """
        Invokes the agent to perform a task.

        Args:
            node (GenAINodeCustom): The node that is invoking the agent.
            input (dict): The input data for the agent.
            gen_ai_memory (GenAIMemory): The memory object for the agent.

        Returns:
            str: The final answer from the agent.
        """
        # TODO: remove node option since may be duplicated with self
        input = self.__check_vars_autofill(input, gen_ai_memory)

        agent_scratchpad = ""
        final_answer = ""
        node_output = None
        for _ in range(self.maximum_iterations):

            input["agent_scratchpad"] = agent_scratchpad

            with warnings.catch_warnings():
                # TODO: check why exists a warning of prompt for bedrock even if Human: Assistant: exist with multiple instructions
                warnings.simplefilter("ignore")

                node_output = self.node_agent_feed_forward.invoke(input, gen_ai_memory)[
                    self.node_agent_feed_forward.output_key
                ]

            if self.platform.value == EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK.value:

                # tools_map
                action, action_input = GenAIAgentTool.extract_action_and_input(
                    node_output
                )
                tool_name, selected_tool = self.find_tool(action)

                if selected_tool is None:

                    if self.func_error_handling is None:
                        raise Exception(f"Error: {node_output}")
                    else:
                        Logging.log(
                            f"Error: {node_output}. Calling func_error_handling for final answer",
                            self.verbose_level,
                            EnumLogs.LOG_LEVEL_ERROR,
                        )

                        action, action_input = self.func_error_handling(
                            {
                                "obj": self,
                                "input": input,
                                "agent_scratchpad": agent_scratchpad,
                                "node_output": node_output,
                            }
                        )
                        tool_name, selected_tool = self.find_tool(action)
                        assert selected_tool is not None

                if tool_name == GenAIAgentTool._normalize_tool_name(
                    GenAIAgentTool.get_tool_human().name
                ):
                    Logging.log(
                        f"TOOL {tool_name}, with: {node_output}",
                        self.verbose_level,
                        EnumLogs.LOG_LEVEL_DEBUG,
                    )
                    final_answer = action_input
                    break

                else:
                    if isinstance(action_input, str):
                        action_input = [action_input]

                    if len(action_input) == 0:
                        raise Exception(
                            f"Error, can't parse: {node_output}. Scratch pad: {agent_scratchpad}"
                        )

                    observation = selected_tool(action_input[-1])
                    agent_scratchpad += f"{node_output} Observation: {observation}. "

                    node_output_clean = node_output.replace("\n", " #")
                    Logging.log(f"AGENT {node_output_clean}", self.verbose_level)
                    Logging.log(
                        f"TOOL OBSERVATION: {tool_name}('{action_input[-1]}') = '{observation}'",
                        self.verbose_level,
                    )

            elif self.platform.value == EnumGenAIPlatforms.PLATFORM_OPENAI.value:
                raise NotImplementedError("OpenAI support not implemented")

        if len(final_answer) == 0:
            final_answer = node_output

        if isinstance(final_answer, list) and len(final_answer) == 1:
            final_answer = final_answer[-1]

        if isinstance(final_answer, str):
            final_answer = (
                final_answer.replace("Response To Human:", "").replace('"', "").strip()
            )

        agent_scratchpad += f"Final Answer: {final_answer}"
        Logging.log(
            f"agent_scratchpad: \n'''\n{agent_scratchpad}'''",
            self.verbose_level,
            EnumLogs.LOG_LEVEL_DEBUG,
        )

        self.state["last_input"] = input
        self.state["last_invoke"] = final_answer

        return self.state["last_invoke"]

    def __check_vars_autofill(self, input: dict, gen_ai_memory: GenAIMemory):
        """
        Checks and fills in the variables in the agent's input based on the chat history injection mode.

        Args:
            input (dict): The input data for the agent.
            gen_ai_memory (GenAIMemory): The memory object for the agent.

        Returns:
            dict: The updated input data with the variables filled in.
        """
        Logging.log(
            f"DEBUG: chat history {str(gen_ai_memory.get_chat_history())}",
            self.verbose_level,
            EnumLogs.LOG_LEVEL_DEBUG,
        )

        if (
            self.chat_history_injection.value == EnumGenAIChatHistoryInjection.INJECTION_MODE_FULL.value
        ):

            input[self.vars_autofill["chat_history"]] = str(
                gen_ai_memory.get_chat_history()
            )

        elif (
            self.chat_history_injection.value == EnumGenAIChatHistoryInjection.INJECTION_MODE_RELATIVE_TO_NAME.value
        ):
            raise NotImplementedError(f"Not implemented: {self.chat_history_injection}")

        elif (
            self.chat_history_injection.value == EnumGenAIChatHistoryInjection.INJECTION_MODE_NOTHING.value
        ):
            input[self.vars_autofill["chat_history"]] = ""

        return input
