from enum import Enum
from langchain.memory import ConversationBufferMemory, SimpleMemory
from datetime import datetime as datetime_genai

class EnumMemoryType(Enum):
    MEMORY_CHAT_BUFFER = "CHAT_BUFFER_MEMORY"
    MEMORY_SIMPLE_MEMORY = "SIMPLE_MEMORY"

class GenAIMemory:

    def __init__(self, memory_type:EnumMemoryType, ai_prefix, human_prefix) -> None:
        
        self.memory_type = memory_type
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        
        if memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:
            
            self.memory = ConversationBufferMemory(
                ai_prefix=self.ai_prefix,
                human_prefix=self.human_prefix
            )

        elif memory_type == EnumMemoryType.MEMORY_SIMPLE_MEMORY:
            self.memory = SimpleMemory(memories={"datetime_creation": str(datetime_genai.utcnow()), "registry": []})
        else:
            raise Exception("NOT IMPLEMENTED")
            
    def add_registry(self, registry, prefix):
        if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:
            
            if prefix == self.ai_prefix:
                self.memory.chat_memory.add_ai_message(registry)
            else:
                self.memory.chat_memory.add_user_message(registry)

        elif self.memory_type == EnumMemoryType.MEMORY_SIMPLE_MEMORY:
            self.memory.memories["registry"].append((prefix, registry))
        else:
            raise Exception("NOT IMPLEMENTED")
        
    def get_raw_memory(self):
        raw = ""
        if self.memory_type == EnumMemoryType.MEMORY_CHAT_BUFFER:
            
            messages = self.memory.chat_memory.messages

            for message in messages:

                if "messages.human" in str(type(message)):
                    raw += f"{self.human_prefix}: {message.content}\n"
                else:
                    raw += f"{self.ai_prefix}: {message.content}\n"

        elif self.memory_type == EnumMemoryType.MEMORY_SIMPLE_MEMORY:
            raw += str(self.memory.memories["registry"])
        else:
            raise Exception("NOT IMPLEMENTED")
        
        return raw.strip()
