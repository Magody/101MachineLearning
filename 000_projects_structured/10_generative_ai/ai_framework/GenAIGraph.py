import operator
from typing import get_args
from ai_framework.GenAIMemory import GenAIMemory, EnumMemoryType
from ai_framework.GenAINode import GenAINode
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAINode import GenAINodeControl, EnumIdNodeControl

from ai_framework.LoggingAndTelemetry import EnumLogs, Logging


class GenAIGraph:
    """
    The GenAIGraph class is responsible for managing the flow of a conversational AI system.
    It provides methods for adding nodes, edges, and conditional edges to the graph, as well as
    for running the graph and updating the state.
    """

    EDGE_UNI_DIRECTION = "forward"
    EDGE_MULTI_DIRECTION = "multi-directional"

    def __init__(
        self,
        state_definition,
        gen_ai_memory: GenAIMemory = None,
        ai_prefix="Assistant",
        human_prefix="Human",
        max_iteration_loop: int = 10,
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        final_output_key="final_output",
        final_output_key_bot="final_output_bot",
    ):
        """
        Initializes the GenAIGraph object with the given parameters.

        Args:
            state_definition (object): The definition of the state for the AI system.
            gen_ai_memory (GenAIMemory, optional): The memory object for the AI system. Defaults to None.
            ai_prefix (str, optional): The prefix for the AI agent. Defaults to "Assistant".
            human_prefix (str, optional): The prefix for the human user. Defaults to "Human".
            max_iteration_loop (int, optional): The maximum number of iterations for the graph. Defaults to 10.
            verbose_level (EnumLogs, optional): The level of verbosity for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.
            final_output_key (str, optional): The key for the final output of the AI system. Defaults to "final_output".
            final_output_key_bot (str, optional): The key for the final output of the bot. Defaults to "final_output_bot".
        """
        self.state_definition = state_definition
        self.reset_state()

        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.gen_ai_memory = gen_ai_memory
        self.max_iteration_loop = max_iteration_loop
        self.verbose_level = verbose_level
        self.final_output_key = final_output_key
        self.final_output_key_bot = final_output_key_bot
        self.final_natural_language_key = None  # key of xml or json format

        self.state_operation = dict()

        for key, annotations in self.state_definition.__annotations__.items():
            for annotation in get_args(annotations):
                if isinstance(annotation, type(operator.add)):
                    self.state_operation[key] = annotation

        self.node_start = GenAINodeControl(EnumIdNodeControl.START, self.verbose_level)
        self.node_end = GenAINodeControl(EnumIdNodeControl.END, self.verbose_level)

        self.nodes = {}
        self.add_node(self.node_start)
        self.add_node(self.node_end)

        self.workflow: dict = {}

        if self.gen_ai_memory is None:
            if self.is_chat_model:
                self.gen_ai_memory = GenAIMemory(
                    GenAIMemory.EnumMemoryType.MEMORY_CHAT_BUFFER,
                    ai_prefix,
                    human_prefix,
                )
            else:
                self.gen_ai_memory = GenAIMemory(
                    EnumMemoryType.MEMORY_SIMPLE_MEMORY, ai_prefix, human_prefix
                )

    def reset_state(self):
        """
        Resets the state of the AI system to the initial state.
        """
        self.state = self.state_definition()

    def update_state(self, state_partial: dict):
        """
        Updates the state of the AI system with the given partial state.

        Args:
            state_partial (dict): The partial state to be updated.
        """
        for key, value in state_partial.items():
            if key not in self.state_definition.__annotations__:
                continue

            if key in self.state_operation:
                self.state[key] = self.state_operation[key](
                    self.state.get(key, []), value
                )
            else:
                self.state[key] = value

    def get_input_variables(self):
        """
        Retrieves the input variables for the current node.

        Returns:
            list: The list of input variables.
        """
        input_variables = self.node_entry.gen_ai_prompt.input_variables
        final_input_variables = []
        exclude_variables = []

        if isinstance(self.node_entry, GenAINodeChain):
            for key in self.node_entry.vars_autofill.values():
                exclude_variables.append(key)

        for input_var in input_variables:
            if input_var not in exclude_variables:
                final_input_variables.append(input_var)

        return final_input_variables

    def add_node(self, node: GenAINode):
        """
        Adds a node to the graph.

        Args:
            node (GenAINode): The node to be added.
        """
        self.nodes[node.id] = node

    def add_edge(self, node_start_edge: GenAINode, node_end_edge: GenAINode):
        """
        Adds an edge between two nodes in the graph.

        Args:
            node_start_edge (GenAINode): The starting node of the edge.
            node_end_edge (GenAINode): The ending node of the edge.
        """

        if node_start_edge.id not in self.nodes:
            self.add_node(node_start_edge)

        if node_end_edge.id not in self.nodes:
            self.add_node(node_end_edge)

        if node_end_edge.id == self.node_end.id:
            if node_start_edge.gen_ai_output_parser is not None:
                self.final_natural_language_key = (
                    node_start_edge.gen_ai_output_parser.natural_language_key
                )

        self.workflow[node_start_edge.id] = {
            "func": lambda graph_object: node_end_edge.id,
            "edge_type": GenAIGraph.EDGE_UNI_DIRECTION,
        }

    def add_conditional_edge(
        self, node_start_edge: GenAINode, func_condition, connections: list
    ):
        """
        Adds a conditional edge between a starting node and a list of possible ending nodes.

        Args:
            node_start_edge (GenAINode): The starting node of the edge.
            func_condition (callable): The function that determines the ending node.
            connections (list): The list of possible ending nodes.
        """

        if node_start_edge.id not in self.nodes:
            self.add_node(node_start_edge)

        self.workflow[node_start_edge.id] = {
            "func": func_condition,
            "edge_type": GenAIGraph.EDGE_MULTI_DIRECTION,
            "connections": connections,
        }

        if self.node_end.id in connections:
            # Same end key could be the another expected in end node
            if node_start_edge.gen_ai_output_parser is not None:
                self.final_natural_language_key = (
                    node_start_edge.gen_ai_output_parser.natural_language_key
                )

    def set_entry_point(self, node_entry: GenAINode):
        """
        Sets the entry point of the graph.

        Args:
            node_entry (GenAINode): The entry point node.
        """
        self.add_edge(self.node_start, node_entry)

    def __str__(self):
        """
        Returns a string representation of the graph.

        Returns:
            str: The string representation of the graph.
        """
        
        max_iter = 100
        mini_graph_to = dict()
        mini_graph_inverse = dict()
        node_seen = set()

        id_nodes = [self.node_start.id]
        output = ""

        while len(id_nodes) > 0 and max_iter > 0:
            id_node = id_nodes.pop(0)

            if id_node not in self.workflow or id_node == self.node_end.id:
                continue

            if id_node not in mini_graph_to:
                mini_graph_to[id_node] = []

            workflow_node = self.workflow[id_node]

            if workflow_node["edge_type"] == GenAIGraph.EDGE_UNI_DIRECTION:
                node_next = workflow_node["func"](self)
                mini_graph_to[id_node].append(node_next)

                if node_next not in mini_graph_inverse:
                    mini_graph_inverse[node_next] = []

                mini_graph_inverse[node_next].append(id_node)

                output += f"({id_node}) -> ({node_next}),"

                if node_next in node_seen:
                    continue
                node_seen.add(node_next)

                if node_next not in id_nodes:
                    id_nodes.append(node_next)

            elif workflow_node["edge_type"] == GenAIGraph.EDGE_MULTI_DIRECTION:
                nodes_next = workflow_node["connections"]
                for id_node_connection in nodes_next:
                    mini_graph_to[id_node].append(id_node_connection)

                    if id_node_connection in node_seen:
                        continue
                    node_seen.add(id_node_connection)

                    if id_node_connection not in id_nodes:
                        id_nodes.append(id_node_connection)

                    if id_node_connection not in mini_graph_inverse:
                        mini_graph_inverse[id_node_connection] = []
                    mini_graph_inverse[id_node_connection].append(id_node)

                    output += f"({id_node}) -> ({id_node_connection}),"
            max_iter -= 1

        if max_iter == 0:
            raise Exception("Unsupported node connections")

        return output[:-1]

    def run(
        self,
        input: dict,
        save_in_memory=True,
        user_natural_language_input="user_input",
    ):
        """
        Runs the graph with the given input and saves the output in memory.

        Args:
            input (dict): The input dictionary.
            save_in_memory (bool, optional): Whether to save the output in memory. Defaults to True.
            user_natural_language_input (str, optional): The key for the user's natural language input. Defaults to "user_input".

        Returns:
            dict: The output dictionary.
        """

        if len(self.workflow) == 0 or self.node_start.id not in self.workflow:
            raise Exception("BadConfiguration")

        Logging.log(
            f"Invocando grafo desde {self.workflow[self.node_start.id]['func'](self)} con {input}",
            self.verbose_level,
            EnumLogs.LOG_LEVEL_DEBUG,
        )

        iteration_count = 1
        iteration_max = max(self.max_iteration_loop, len(self.nodes) + 2)

        id_actual_node = self.node_start.id
        id_end_node = self.node_end.id
        output = input
        self.update_state(input)

        while iteration_count < iteration_max and id_actual_node != id_end_node:
            # TODO: use state instead of output
            output_node = self.nodes[id_actual_node].invoke(output, self.gen_ai_memory)

            output = {**output, **output_node}

            if id_actual_node in self.workflow:
                # TODO: Bug when state change and not reflect in inmediate output
                # print("SAVING", output_node)
                self.update_state(output)
                id_next_node = self.workflow[id_actual_node]["func"](self)
                Logging.log(
                    f"FLOW: actual={id_actual_node} -> next={id_next_node}",
                    self.verbose_level,
                )
                assert id_next_node in self.nodes
                id_actual_node = id_next_node
            else:
                break

            output = {**output, **self.state}

            iteration_count += 1

        if save_in_memory:
            self.gen_ai_memory.add_chat_registry(
                input[user_natural_language_input], self.human_prefix
            )

            try:
                self.gen_ai_memory.add_chat_registry(
                    output[self.final_output_key], self.ai_prefix
                )
            except Exception as error:
                Logging.log(f"\n\n\nERROR in message parsing: {error}\n\n\n", self.verbose_level)
                if len(output.keys()) == 1:
                    self.gen_ai_memory.add_chat_registry(
                        str(output[list(output.keys())[0]]), self.ai_prefix
                    )
                else:
                    self.gen_ai_memory.add_chat_registry(str(output), self.ai_prefix)

        return output
