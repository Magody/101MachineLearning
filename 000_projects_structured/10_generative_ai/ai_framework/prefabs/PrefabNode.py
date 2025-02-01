from abc import ABC
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAIGraph import GenAIGraph
from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging


class PrefabNode(ABC):
    """
    The PrefabNode class is an abstract base class that provides a foundation for creating nodes in an AI framework.
    It handles the initialization of the node, including its ID, the GenAILLM instance, the nodes dictionary, input maps,
    verbose level, and the next node ID (if applicable). The class also provides methods for invoking the node, handling
    edge connections, and retrieving the IDs of the nodes.
    """

    def __init__(
        self,
        id,
        gen_ai_llm: GenAILLM,
        nodes,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        node_next: str = None
    ) -> None:
        """
        Initializes the PrefabNode instance.

        Args:
            id (str): The unique identifier of the node.
            gen_ai_llm (GenAILLM): The GenAILLM instance used by the node.
            nodes (dict): A dictionary of node IDs and their corresponding next node IDs.
            input_maps (dict, optional): A dictionary mapping input keys to their corresponding names. Defaults to
                {"user_input": "user_input", "final_output": "final_output", "final_output_bot": "final_output_bot"}.
            verbose_level (EnumLogs, optional): The verbose level for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.
            node_next (str, optional): The ID of the next node. Defaults to None.
        """
        self.id = id
        self.node: GenAINodeChain = None
        self.nodes: dict = nodes
        self.input_maps = input_maps
        self.verbose_level = verbose_level
        self.gen_ai_llm = gen_ai_llm

        if self.nodes is None and node_next is not None:
            self.nodes = {
                "node_id_next": node_next,
            }

    def func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        Invokes the node and updates the shared state with the result.

        Args:
            node (PrefabNode): The node to be invoked.
            shared_state (dict): The shared state dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory instance.

        Returns:
            dict: The updated shared state dictionary.
        """
        # TODO: remove need of 'node' in parameter since in new architecture self is enough
        Logging.log(f"RUNNING: {self.node.id}", self.node.verbose_level)

        chain_result = self.node.get_chain_result(shared_state, gen_ai_memory)

        return {
            **shared_state,
            "intermediate_steps_bp": [(self.node.id, "default")],
            **chain_result,
            # TODO: Bug in GenAINodeChain, who extract data from node_outcome and edge changing state not reflect due to override
            "node_outcome": chain_result,
        }

    def func_edge_connection(self, graph_object: GenAIGraph):
        """
        Handles the edge connection logic for the node.

        Args:
            graph_object (GenAIGraph): The GenAIGraph instance.

        Returns:
            str: The ID of the next node to be invoked.
        """
        last_output = ""
        if graph_object.final_output_key_bot in graph_object.state:
            last_output = graph_object.state[graph_object.final_output_key_bot]
        Logging.log(
            f"EDGE function, last output: {last_output}",
            self.node.verbose_level,
            minimum_verbose_level=EnumLogs.LOG_LEVEL_DEBUG,
        )

        # Default next: node_id_next key to invoke the value
        return self.nodes["node_id_next"]

    def invoke(self, input: dict, gen_ai_memory: GenAIMemory):
        """
        Invokes the node with the given input and GenAIMemory instance.

        Args:
            input (dict): The input dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory instance.

        Returns:
            dict or None: The result of the node invocation, or None if the node is not initialized.
        """
        if self.node is None:
            Logging.log(
                "self.node is None. First instantiate it", EnumLogs.LOG_LEVEL_ERROR
            )
            return None

        # Bridge
        return self.node.invoke(input, gen_ai_memory)

    def get_nodes_ids(self):
        """
        Retrieves the IDs of the nodes.

        Returns:
            list: A list of node IDs.
        """
        return self.nodes.values()
