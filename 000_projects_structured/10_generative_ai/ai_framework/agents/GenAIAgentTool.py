import re


class GenAIAgentTool:
    """
    A class representing a tool for a GenAI agent.

    Attributes:
        name (str): The name of the tool.
        description (str): The description of the tool.
        func (callable): The function associated with the tool.
        return_direct (bool): Indicates whether the tool should return the input directly.
        agent_tool_human (GenAIAgentTool): A singleton instance of the "Response To Human" tool.

    Methods:
        get_tool_human(): Returns the singleton instance of the "Response To Human" tool.
        get_tools_definitions(tools: list, format: str = "common"): Returns the definitions of the given tools in the specified format.
        get_tools_names(tools: list): Returns the names of the given tools.
        extract_action_and_input(text: str): Extracts the action and input from the given text.
        _normalize_tool_name(tool_name: str): Normalizes the tool name by removing spaces and converting to lowercase.
    """

    agent_tool_human = None

    def __init__(self, name: str, description: str, func, return_direct=False) -> None:
        """
        Initializes a GenAIAgentTool instance.

        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.
            func (callable): The function associated with the tool.
            return_direct (bool, optional): Indicates whether the tool should return the input directly. Defaults to False.
        """
        self.name = name
        self.description = description
        self.func = func
        self.return_direct = return_direct

    @staticmethod
    def get_tool_human():
        """
        Returns the singleton instance of the "Response To Human" tool.

        Returns:
            GenAIAgentTool: The singleton instance of the "Response To Human" tool.
        """
        if GenAIAgentTool.agent_tool_human is None:
            # Singleton Pattern Design
            GenAIAgentTool.agent_tool_human = GenAIAgentTool(
                "Response To Human",
                "When you need to respond to the human you are talking to.",
                lambda action_input: action_input,
            )
        return GenAIAgentTool.agent_tool_human

    @staticmethod
    def get_tools_definitions(tools: list, format="common"):
        """
        Returns the definitions of the given tools in the specified format.

        Args:
            tools (list): A list of GenAIAgentTool instances.
            format (str, optional): The format of the tool definitions. Can be "common" or "xml". Defaults to "common".

        Returns:
            str: The tool definitions in the specified format.
        """
        # TODO: create enum for formats
        tools_definitions = ""
        for tool in tools:
            if isinstance(tool, GenAIAgentTool):
                if format == "common":
                    tools_definitions += f"{tool.name}: {tool.description}\n"
                elif format == "xml":
                    tools_definitions += f"<tool><name>{tool.name}</name><description>{tool.description}</description></tool>\n"

        return tools_definitions

    @staticmethod
    def get_tools_names(tools: list):
        """
        Returns the names of the given tools.

        Args:
            tools (list): A list of GenAIAgentTool instances.

        Returns:
            str: The names of the tools enclosed in parentheses.
        """
        # TODO: create enum for formats
        tools_names = ""
        for tool in tools:
            if isinstance(tool, GenAIAgentTool):
                tools_names += f"{tool.name}, "

        tools_names = tools_names[:-2]
        return f"({tools_names})"

    @staticmethod
    def extract_action_and_input(text):
        """
        Extracts the action and input from the given text.

        Args:
            text (str): The text containing the action and input.

        Returns:
            tuple: A tuple containing the action and input.
        """
        action_pattern = r"Action: (.+)\n"
        input_pattern = r"Action Input: \"?(.+)\"?"
        action = re.findall(action_pattern, text)
        action_input = re.findall(input_pattern, text)
        return action, action_input

    @staticmethod
    def _normalize_tool_name(tool_name):
        """
        Normalizes the tool name by removing spaces and converting to lowercase.

        Args:
            tool_name (str): The tool name to be normalized.

        Returns:
            str: The normalized tool name.
        """
        return re.sub(r"\s", "", tool_name).lower()
