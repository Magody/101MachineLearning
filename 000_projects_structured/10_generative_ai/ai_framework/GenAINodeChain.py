from enum import Enum
from langchain.chains import SequentialChain, LLMChain
from ai_framework.GenAILLM import EnumCostMode, GenAILLM
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINode import GenAINode
from ai_framework.GenAIOutputParser import GenAIOutputParser
from ai_framework.GenAIPrompt import GenAIPrompt
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging
from datetime import datetime as datetime_node


class EnumGenAINodeChainType(Enum):
    """Enum for different types of GenAINodeChain"""
    CHAIN_TYPE_COMMON = "LLMChain"
    CHAIN_TYPE_GUARDRAIL = "Guardrail"
    CHAIN_TYPE_CLASSIFICATION = "Classification"
    CHAIN_TYPE_COMMON_PLUS_RAG = "LLMChainRAG"


class GenAINodeChain(GenAINode):
    """
    A class that represents a GenAINodeChain, which is a subclass of GenAINode.

    Attributes:
        chain_type (EnumGenAINodeChainType): The type of the chain.
        chain (LLMChain): The LLMChain object used for the chain.

    Methods:
        func_invoke_chain_default(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
            Invokes the default chain function.
        __check_vars_autofill(self, input: dict, gen_ai_memory: GenAIMemory):
            Checks and fills the variables in the input dictionary.
        extract_final_output(self, chain_result):
            Extracts the final output from the chain result.
        inject_node_metadata(self, chain_result):
            Injects node metadata into the chain result.
        pos_process_chain_result(self, chain_result, gen_ai_memory: GenAIMemory):
            Post-processes the chain result.
        get_chain_result(self, input: dict, gen_ai_memory: GenAIMemory, intent_chat_history: str = None):
            Gets the chain result.
        invoke(self, input: dict, gen_ai_memory: GenAIMemory):
            Invokes the chain.

    Static Methods:
        create_sequential_chain(memory, chains, input_variables, partial_variables, output_variables=["final_output"], verbose=False):
            Creates a sequential chain.
    """

    def __init__(
        self,
        id: str,
        gen_ai_llm: GenAILLM,
        gen_ai_prompt: GenAIPrompt,
        gen_ai_output_parser: GenAIOutputParser = None,
        func_invoke=None,
        chain_type: EnumGenAINodeChainType = EnumGenAINodeChainType.CHAIN_TYPE_COMMON,
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
        """
        Initializes a GenAINodeChain object.

        Args:
            id (str): The ID of the node.
            gen_ai_llm (GenAILLM): The GenAILLM object.
            gen_ai_prompt (GenAIPrompt): The GenAIPrompt object.
            gen_ai_output_parser (GenAIOutputParser, optional): The GenAIOutputParser object.
            func_invoke (callable, optional): The function to invoke.
            chain_type (EnumGenAINodeChainType, optional): The type of the chain.
            verbose_level (EnumLogs, optional): The verbose level.
            inject_memory_common (bool, optional): Whether to inject common memory.
            inject_memory_chat (bool, optional): Whether to inject chat history.
            inject_memory_intent_chat (bool, optional): Whether to inject intent chat history.
            vars_autofill (dict, optional): The dictionary of variables to autofill.
            output_key (str, optional): The output key.
        """
        if func_invoke is None:
            func_invoke = self.func_invoke_chain_default

        super().__init__(
            id,
            gen_ai_llm,
            gen_ai_prompt,
            gen_ai_output_parser,
            func_invoke,
            verbose_level,
            inject_memory_common,
            inject_memory_chat,
            inject_memory_intent_chat,
            vars_autofill,
            output_key,
        )
        self.chain_type = chain_type

        self.chain = LLMChain(
            llm=gen_ai_llm.llm_model,
            prompt=gen_ai_prompt.prompt_template,
            output_key=self.output_key,
            verbose=gen_ai_llm.verbose,
        )

    def func_invoke_chain_default(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        Invokes the default chain function.

        Args:
            node (GenAINode): The GenAINode object.
            shared_state (dict): The shared state dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory object.

        Returns:
            dict: The updated shared state dictionary.
        """
        # TODO: remove need of 'node' in parameter since in new architecture self is enough. Also in prefabs refactor to not duplicate code
        Logging.log(f"RUNNING: {self.id}", self.verbose_level)

        chain_result = self.get_chain_result(shared_state, gen_ai_memory)

        return {
            **shared_state,
            "intermediate_steps_bp": [(self.id, "default")],
            **chain_result,
            # TODO: Bug in GenAINodeChain, who extract data from node_outcome and edge changing state not reflect due to override
            "node_outcome": chain_result,
        }

    def __check_vars_autofill(self, input: dict, gen_ai_memory: GenAIMemory):
        """
        Checks and fills the variables in the input dictionary.

        Args:
            input (dict): The input dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory object.

        Returns:
            dict: The updated input dictionary.
        """
        if self.inject_memory_common:
            input[self.vars_autofill["context"]] = str(
                gen_ai_memory.get_common_memories()
            )

        if self.inject_memory_chat:
            input[self.vars_autofill["chat_history"]] = str(
                gen_ai_memory.get_chat_history()
            )

        if (
            self.vars_autofill["format_instructions"]
            in self.gen_ai_prompt.input_variables
        ):

            if self.gen_ai_output_parser is not None:
                input[self.vars_autofill["format_instructions"]] = (
                    self.gen_ai_output_parser.get_instructions()
                )
            elif not self.vars_autofill["format_instructions"] in input:
                input[self.vars_autofill["format_instructions"]] = ""
        return input

    def extract_final_output(self, chain_result):
        """
        Extracts the final output from the chain result.

        Args:
            chain_result (dict): The chain result.

        Returns:
            tuple: The updated chain result and the final output.
        """

        if self.gen_ai_output_parser is not None:

            if "final-answer" in chain_result[self.output_key] and isinstance(
                chain_result[self.output_key], dict
            ):

                if (
                    "#text"
                    in chain_result[self.output_key]["final-answer"][
                        self.gen_ai_output_parser.natural_language_key
                    ]
                ):
                    final_output = chain_result[self.output_key]["final-answer"][
                        self.gen_ai_output_parser.natural_language_key
                    ]["#text"]
                else:
                    final_output = chain_result[self.output_key]["final-answer"][
                        self.gen_ai_output_parser.natural_language_key
                    ]

                chain_result[self.output_key] = final_output
            else:
                final_output = chain_result[self.output_key]
        else:
            final_output = chain_result[self.output_key]

        return chain_result, final_output

    def inject_node_metadata(self, chain_result):
        """
        Injects node metadata into the chain result.

        Args:
            chain_result (dict): The chain result.

        Returns:
            dict: The updated chain result.
        """
        chain_result["id"] = self.id
        return chain_result

    def pos_process_chain_result(self, chain_result, gen_ai_memory: GenAIMemory):
        """
        Post-processes the chain result.

        Args:
            chain_result (dict): The chain result.
            gen_ai_memory (GenAIMemory): The GenAIMemory object.

        Returns:
            dict: The updated chain result.
        """
        chain_result, final_output = self.extract_final_output(chain_result)
        chain_result = self.inject_node_metadata(chain_result)

        if self.chain_type not in (
            EnumGenAINodeChainType.CHAIN_TYPE_GUARDRAIL,
            EnumGenAINodeChainType.CHAIN_TYPE_CLASSIFICATION,
        ):
            # Final output is only message in NLP
            chain_result["final_output"] = final_output

        if self.gen_ai_output_parser is not None:
            chain_result[self.output_key] = self.gen_ai_output_parser.parse(
                chain_result[self.output_key]
            )

        if self.id not in gen_ai_memory.memory.memories:
            gen_ai_memory.memory.memories[self.id] = []
        gen_ai_memory.add_common_registry(self.id, final_output)

        # The real final output even if is inner thought of bot
        chain_result["final_output_bot"] = final_output
        return chain_result

    def get_chain_result(
        self, input: dict, gen_ai_memory: GenAIMemory, intent_chat_history: str = None
    ):
        """
        Gets the chain result.

        Args:
            input (dict): The input dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory object.
            intent_chat_history (str, optional): The intent chat history. Defaults to None.

        Returns:
            dict: The chain result.
        """
        if self.inject_memory_intent_chat:
            input[self.vars_autofill["chat_history_intent"]] = str(intent_chat_history)

        if "node_outcome" in input:
            input = {**input, **input["node_outcome"]}

        try:
            chain_result = self.chain.invoke(input)
        except Exception as error:
            Logging.log(f"Error in GenAINodeChain.get_chain_result: {error}. Trying to use 'user_input' or default key", self.verbose_level, EnumLogs.LOG_LEVEL_ERROR)
            input_keys = input.keys()
            if len(input_keys) == 1:
                chain_result = dict()
                chain_result[self.output_key] = self.gen_ai_llm.invoke(input[input_keys[0]])
            else:
                raise error

        return self.pos_process_chain_result(chain_result, gen_ai_memory)

    def invoke(self, input: dict, gen_ai_memory: GenAIMemory):
        """
        Invokes the chain.

        Args:
            input (dict): The input dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory object.

        Returns:
            dict: The chain result.
        """
        time_start = datetime_node.now()

        input = self.__check_vars_autofill(input, gen_ai_memory)
        if self.func_invoke is None:
            res_invoke = self.get_chain_result(input, gen_ai_memory)
        else:
            res_invoke = self.func_invoke(self, input, gen_ai_memory)
        # User input is not the same as the prompt sent to LLM, input may include rag, chat history, etc
        # TODO: prefilter the right partials // completes for format
        self.input_formated_prompt = self.gen_ai_prompt.prompt_template.format(**{**input, **res_invoke})

        self.input_tokens = self.gen_ai_llm.calculate_tokens(self.input_formated_prompt)
        self.input_expected_cost = self.gen_ai_llm.calculate_cost(
            self.input_tokens + GenAILLM.DELTA_EXTRA_TOKENS_FOR_COST,
            EnumCostMode.COST_INPUT_TOKENS,
        )

        self.output_tokens = self.gen_ai_llm.calculate_tokens(
            res_invoke[self.output_key]
        )
        self.output_expected_cost = self.gen_ai_llm.calculate_cost(
            self.output_tokens + GenAILLM.DELTA_EXTRA_TOKENS_FOR_COST,
            EnumCostMode.COST_OUTPUT_TOKENS,
        )
        time_end = datetime_node.now()

        time_difference_ms = round((time_end - time_start).microseconds / 1000, 2)

        self.log_and_save(time_difference_ms, absolute_node_input=self.input_formated_prompt)
        return res_invoke

    @staticmethod
    def create_sequential_chain(
        memory,
        chains,
        input_variables,
        partial_variables,
        output_variables=["final_output"],
        verbose=False,
    ):
        """
        Creates a sequential chain.

        Args:
            memory (GenAIMemory): The GenAIMemory object.
            chains (list): The list of chains.
            input_variables (list): The list of input variables.
            partial_variables (list): The list of partial variables.
            output_variables (list, optional): The list of output variables.
            verbose (bool, optional): Whether to enable verbose mode.

        Returns:
            SequentialChain: The created sequential chain.
        """
        return SequentialChain(
            memory=memory,
            chains=chains,
            input_variables=input_variables,
            output_variables=output_variables,
            partial_variables=partial_variables,
            verbose=verbose,
        )
