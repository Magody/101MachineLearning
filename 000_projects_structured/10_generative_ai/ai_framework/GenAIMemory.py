from enum import Enum
from langchain.memory import ConversationBufferMemory, SimpleMemory
from datetime import datetime as datetime_genai
import pytz


class EnumMemoryType(Enum):
    """Enum to define the type of memory to be used"""
    MEMORY_CHAT_BUFFER = "CHAT_BUFFER_MEMORY"
    MEMORY_SIMPLE_MEMORY = "SIMPLE_MEMORY"


class GenAIMemory:
    """
    Component responsible for storing a conversation history and variables in a memory dictionary.

    Attributes:
        INDEX (str): Key for the index of the chat registry.
        PREFIX (str): Key for the prefix of the chat registry.
        CONTENT (str): Key for the content of the chat registry.
        memory_type (EnumMemoryType): The type of memory to be used.
        ai_prefix (str): The prefix for the AI messages.
        human_prefix (str): The prefix for the human messages.
        memory_key (str): The key for the memory.
        input_key (str): The key for the input.
        memory_chat (ConversationBufferMemory or None): The chat memory.
        maximum_memory_keys (int): The maximum number of keys to store in the memory.
    """

    INDEX = "indice"
    PREFIX = "prefijo"
    CONTENT = "contenido"

    def __init__(
        self,
        ai_prefix="Assistant",
        human_prefix="Human",
        memory_key='history', 
        input_key='input',
        memory_type: EnumMemoryType = EnumMemoryType.MEMORY_CHAT_BUFFER,
        maximum_memory_keys=10,
    ) -> None:
        """
        Initialize the GenAIMemory object.

        Args:
            ai_prefix (str, optional): The prefix for the AI messages. Defaults to "Assistant".
            human_prefix (str, optional): The prefix for the human messages. Defaults to "Human".
            memory_key (str, optional): The key for the memory. Defaults to 'history'.
            input_key (str, optional): The key for the input. Defaults to 'input'.
            memory_type (EnumMemoryType, optional): The type of memory to be used. Defaults to EnumMemoryType.MEMORY_CHAT_BUFFER.
            maximum_memory_keys (int, optional): The maximum number of keys to store in the memory. Defaults to 10.
        """
        self.memory_type = memory_type
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.memory_key = memory_key
        self.input_key = input_key
        self.memory_chat = None
        self.maximum_memory_keys = maximum_memory_keys

        self.reboot()

    def _get_time(self):
        """
        Get the current time in the "America/Guayaquil" timezone.

        Returns:
            datetime: The current time in the "America/Guayaquil" timezone.
        """
        return datetime_genai.now(pytz.timezone("America/Guayaquil"))

    def reboot(self):
        """
        Reboot the memory by initializing the appropriate memory type.
        """
        if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:

            self.memory_chat = ConversationBufferMemory(
                ai_prefix=self.ai_prefix, human_prefix=self.human_prefix, memory_key=self.memory_key, input_key=self.input_key
            )

        self.memory = SimpleMemory(
            memories={
                "datetime_creation": {
                    "value": self._get_time(),
                    "time": self._get_time(),
                },
            }
        )

    def update_memory(self, common_memories: dict, chat_history: list):
        """
        Update the memory with the provided common memories and chat history.

        Args:
            common_memories (dict): A dictionary of common memories to be updated.
            chat_history (list): A list of chat history entries to be added.
        """
        memories_parsed = self.memory.memories

        for key, value in common_memories.items():
            if key not in memories_parsed:
                memories_parsed[key] = value

            print(key, value)

            if key == "datetime_creation":
                memories_parsed[key]["value"] = datetime_genai.strptime(
                    value["value"], "%Y-%m-%d %H:%M:%S.%f%z"
                )
                memories_parsed[key]["time"] = datetime_genai.strptime(
                    value["time"], "%Y-%m-%d %H:%M:%S.%f%z"
                )

            elif "time" in value:
                memories_parsed[key]["time"] = datetime_genai.strptime(
                    value["time"], "%Y-%m-%d %H:%M:%S.%f%z"
                )

        self.memory.memories = memories_parsed

        chat_history.sort(key=lambda s: s[GenAIMemory.INDEX])
        # TODO: limit in saving too
        # chat_history = chat_history[-20:]

        for registry in chat_history:
            self.add_chat_registry(
                registry[GenAIMemory.CONTENT], registry[GenAIMemory.PREFIX]
            )

    def add_chat_registry(self, registry, prefix, source="unknown"):
        """
        Add a chat registry to the memory.

        Args:
            registry (str): The content of the chat registry.
            prefix (str): The prefix of the chat registry.
            source (str, optional): The source of the chat registry. Defaults to "unknown".
        """
        if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:

            if prefix == self.ai_prefix:
                self.memory_chat.chat_memory.add_ai_message(registry)
            else:
                self.memory_chat.chat_memory.add_user_message(registry)

        else:
            if "conversation" not in self.memory.memories:
                self.memory.memories["conversation"] = []

            self.add_common_registry("conversation", (prefix, registry, source))

    def add_common_registry(self, key, value):
        """
        Add a common registry to a general python dictionary.

        Args:
            key (str): The key for the common registry.
            value (any): The value for the common registry.
        """
        if key in self.memory.memories:
            if isinstance(self.memory.memories[key], list):
                self.memory.memories[key].append(value)
                if len(self.memory.memories[key]) > 100:
                    self.memory.memories[key].pop(0)
                return

        self.memory.memories[key] = {"value": value, "time": self._get_time()}

        if len(self.memory.memories) > self.maximum_memory_keys:
            min_time = self._get_time()
            key_delete = key

            for key_in in self.memory.memories.keys():
                memory_object = self.memory.memories[key_in]

                if memory_object["time"] < min_time:
                    key_delete = key_in
                    min_time = memory_object["time"]

            del self.memory.memories[key_delete]

    def get_chat_history(self, return_mode="default"):
        """
        Get the chat history in the desired format.

        Args:
            return_mode (str, optional): The desired format of the chat history. Can be "default", "list", or "dict". Defaults to "default".

        Returns:
            str or list or dict: The chat history in the desired format.
        """

        if "list" in return_mode.lower():
            chat_history_list = []
            if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:
                messages = self.memory_chat.chat_memory.messages

                for i, message in enumerate(messages):

                    if "messages.human" in str(type(message)):
                        chat_history_list.append(
                            (i + 1, self.human_prefix, message.content)
                        )
                    else:
                        chat_history_list.append(
                            (i + 1, self.ai_prefix, message.content)
                        )

            return chat_history_list

        if "dict" in return_mode.lower():
            chat_history_dicts = []

            if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:
                messages = self.memory_chat.chat_memory.messages

                for i, message in enumerate(messages):

                    if "messages.human" in str(type(message)):
                        chat_history_dicts.append(
                            {
                                GenAIMemory.INDEX: i + 1,
                                GenAIMemory.PREFIX: self.human_prefix,
                                GenAIMemory.CONTENT: message.content,
                            }
                        )
                    else:
                        chat_history_dicts.append(
                            {
                                GenAIMemory.INDEX: i + 1,
                                GenAIMemory.PREFIX: self.ai_prefix,
                                GenAIMemory.CONTENT: message.content,
                            }
                        )

            return chat_history_dicts

        else:
            chat_history = ""
            if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:
                messages = self.memory_chat.chat_memory.messages

                for message in messages:

                    if "messages.human" in str(type(message)):
                        chat_history += f"{self.human_prefix}: {message.content}\n"
                    else:
                        chat_history += f"{self.ai_prefix}: {message.content}\n"

            return chat_history.strip()

    def get_common_memories(self):
        """
        Get the common memories.

        Returns:
            dict: The common memories.
        """
        return self.memory.memories

    def __str__(self):
        """
        Get a string representation of the GenAIMemory object.

        Returns:
            str: The string representation of the GenAIMemory object.
        """
        raw = f"<historial-chat>\n{self.get_chat_history()}\n</historial-chat>\n\n"
        raw += f"<memoria>{self.get_common_memories()}</memoria>"

        return raw
