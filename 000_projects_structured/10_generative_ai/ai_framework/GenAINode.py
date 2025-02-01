from abc import ABC
from enum import Enum
from ai_framework.GenAILLM import GenAILLM

from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAIOutputParser import GenAIOutputParser
from ai_framework.GenAIPrompt import GenAIPrompt
from ai_framework.LoggingAndTelemetry import EnumLogs
from datetime import datetime as datetime_node


class GenAINode(ABC):
    """
    The GenAINode class is an abstract base class that represents a node in a generative AI framework.
    It provides a common interface for managing the input, output, and execution of a node.

    Args:
        id (str): The unique identifier of the node.
        gen_ai_llm (GenAILLM): The language model used by the node.
        gen_ai_prompt (GenAIPrompt): The prompt used by the node.
        gen_ai_output_parser (GenAIOutputParser): The output parser used by the node.
        func_invoke (callable, optional): A custom function to be invoked when the node is executed.
        verbose_level (EnumLogs, optional): The verbosity level for logging and telemetry.
        inject_memory_common (bool, optional): Whether to inject common memory into the node.
        inject_memory_chat (bool, optional): Whether to inject chat history into the node.
        inject_memory_intent_chat (bool, optional): Whether to inject intent chat history into the node.
        vars_autofill (dict, optional): A dictionary of variable names to be automatically filled.
        output_key (str, optional): The key to use for the node's output.
    """

    def __init__(
        self,
        id: str,
        gen_ai_llm: GenAILLM,
        gen_ai_prompt: GenAIPrompt,
        gen_ai_output_parser: GenAIOutputParser,
        func_invoke=None,
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        inject_memory_common=False,
        inject_memory_chat=False,
        inject_memory_intent_chat=False,
        vars_autofill: dict = {
            "context": "context",
            "chat_history": "chat_history",
            "chat_history_intent": "chat_history_intent",
            "format_instructions": "format_instructions",
        },
        output_key="auto",
    ):
        self.id = id
        self.gen_ai_llm = gen_ai_llm
        self.gen_ai_prompt = gen_ai_prompt
        self.gen_ai_output_parser = gen_ai_output_parser
        self.func_invoke = func_invoke
        self.verbose_level = verbose_level
        self.inject_memory_common = inject_memory_common
        self.inject_memory_chat = inject_memory_chat
        self.inject_memory_intent_chat = inject_memory_intent_chat
        self.vars_autofill = vars_autofill

        if output_key == "auto":
            self.output_key = f"output_{self.id}"
        else:
            self.output_key = output_key

        self.reset()

    def reset(self):
        """
        Resets the internal state of the node.
        """
        self.input_tokens = None
        self.input_expected_cost = None
        self.output_tokens = None
        self.output_expected_cost = None

        self.absolute_node_input_history = []
        self.status_history = {
            "INPUT_TOKENS": [],
            "OUTPUT_TOKENS": [],
            "TIME": [],
        }

        self.total_input_tokens = 0
        self.total_input_tokens_expected_cost = 0
        self.total_output_tokens = 0
        self.total_output_tokens_expected_cost = 0
        self.total_time = 0

    def get_cost_time_summary(self):
        """
        Returns a summary of the cost and time metrics for the node.
        """
        return f"""
        Total time: {self.total_time} ms
        Total expected cost: {self.total_input_tokens_expected_cost + self.total_output_tokens_expected_cost}
        Total expected INPUT cost: {self.total_input_tokens_expected_cost}
        Last INPUT tokens: {self.input_tokens}. Last expected INPUT cost: {self.input_expected_cost}
        Total expected OUTPUT cost: {self.total_output_tokens_expected_cost}
        Last OUTPUT tokens: {self.output_tokens}. Last expected OUTPUT cost: {self.output_expected_cost}
        """

    def get_cost_time_history(self):
        """
        Returns the history of the cost and time metrics for the node.
        """
        last_input_history = ""
        if len(self.absolute_node_input_history) > 0:
            last_input_history = self.absolute_node_input_history[-1]

        cost_input_output = ""

        i = 0
        j = 0
        while i < len(self.status_history["INPUT_TOKENS"]):

            input_element_count = self.status_history["INPUT_TOKENS"][i]
            output_element_count = self.status_history["OUTPUT_TOKENS"][i]
            input_element_cost = self.status_history["INPUT_TOKENS"][i + 1]
            output_element_cost = self.status_history["OUTPUT_TOKENS"][i + 1]

            element_time = self.status_history["TIME"][j]
            i += 2
            j += 1

            cost_input_output += f"""*****
            input (tokens, cost): ({input_element_count[1]}, {input_element_cost[1]})
            output (tokens, cost): ({output_element_count[1]}, {output_element_cost[1]})
            time: {element_time[1]} [ms]
            """

        return f"{cost_input_output}\nlast_input_history: {last_input_history}"

    def log_and_save(self, time_to_log_ms, absolute_node_input):
        """
        Logs and saves the input, output, and time metrics for the node.

        Args:
            time_to_log_ms (float): The execution time of the node in milliseconds.
            absolute_node_input (str): The absolute input to the node.
        """
        if self.input_tokens is not None:
            self.status_history["INPUT_TOKENS"].append(
                ("INPUT_TOKENS_COUNT", self.input_tokens)
            )
            self.status_history["INPUT_TOKENS"].append(
                ("INPUT_TOKENS_EXPECTED_COST", self.input_expected_cost)
            )

            self.status_history["OUTPUT_TOKENS"].append(
                ("OUTPUT_TOKENS_COUNT", self.output_tokens)
            )
            self.status_history["OUTPUT_TOKENS"].append(
                ("OUTPUT_TOKENS_EXPECTED_COST", self.output_expected_cost)
            )

            self.status_history["TIME"].append(
                ("EXECUTION_TIME", time_to_log_ms)
            )

            if len(self.status_history["INPUT_TOKENS"]) > 100:
                self.status_history["INPUT_TOKENS"].pop(0)
                self.status_history["OUTPUT_TOKENS"].pop(0)
                self.status_history["TIME"].pop(0)

            self.total_input_tokens += self.input_tokens
            self.total_output_tokens += self.output_tokens
            self.total_input_tokens_expected_cost += self.input_expected_cost
            self.total_output_tokens_expected_cost += (
                self.output_expected_cost
            )
            self.total_time += time_to_log_ms

            self.absolute_node_input_history.append(absolute_node_input)

            if len(self.absolute_node_input_history) > 10:
                self.absolute_node_input_history.pop(0)

    def invoke(self, input: dict, component_memory: GenAIMemory = None):
        """
        Invokes the node with the given input and optional component memory.

        Args:
            input (dict): The input to the node.
            component_memory (GenAIMemory, optional): The component memory to be used by the node.

        Returns:
            The result of invoking the node.
        """
        time_start = datetime_node.now()
        res_invoke = None
        if self.func_invoke is not None:
            res_invoke = self.func_invoke(self, input, component_memory)
        else:
            res_invoke = input

        if "absolute_node_input" in input:
            abs_input = input["absolute_node_input"]
        else:
            abs_input = ""
            for value in input.values():
                if isinstance(value, str):
                    abs_input += value
        time_end = datetime_node.now()

        time_difference_ms = round((time_end - time_start).microseconds / 1000, 2)

        self.log_and_save(time_difference_ms, absolute_node_input=abs_input)
        return res_invoke

    def __str__(self):
        """
        Returns a string representation of the node.
        """
        return f"""Nodo id: {self.id}. 
        Costo total input: {self.total_input_tokens_expected_cost}. Costo total output: {self.total_output_tokens_expected_cost}. inject_memory_common: {self.inject_memory_common}.
        inject_memory_chat: {self.inject_memory_chat}. inject_memory_intent_chat: {self.inject_memory_intent_chat}
        """


class EnumIdNodeControl(Enum):
    """
    An enumeration of generic nodes without functionality.
    """
    START = "NODE_CONTROL_START"
    END = "NODE_CONTROL_END"


class GenAINodeControl(GenAINode):
    """
    The GenAINodeControl class represents a control node in the generative AI framework.

    Args:
        id (EnumIdNodeControl): The unique identifier of the control node.
        verbose_level (EnumLogs, optional): The verbosity level for logging and telemetry.
    """

    def __init__(
        self, id: EnumIdNodeControl, verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO
    ):
        super().__init__(id.value, None, None, None, None, verbose_level)


class GenAINodeCustom(GenAINode):
    """
    The GenAINodeCustom class represents a custom node in the generative AI framework.

    Args:
        id (str): The unique identifier of the custom node.
        func_invoke (callable): The custom function to be invoked when the node is executed.
        verbose_level (EnumLogs, optional): The verbosity level for logging and telemetry.
    """
    
    def __init__(
        self, id: str, func_invoke, verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO
    ):
        super().__init__(id, None, None, None, func_invoke, verbose_level)
