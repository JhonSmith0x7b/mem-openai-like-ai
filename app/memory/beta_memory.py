
from mem0 import Memory
import os
from typing import Dict, List


DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID")
DEFAULT_ASSISTANT_ID = os.environ.get("DEFAULT_ASSISTANT_ID")
MEMORY_HISTORY_LIMIT = int(os.environ.get("MEMORY_HISTORY_LIMIT"))
MEMORY_ADD_LIMIT = int(os.environ.get("MEMORY_ADD_LIMIT"))
MEMORY_MODEL = os.environ.get("MEMORY_MODEL")
GRAPH_ON = os.environ.get("GRAPH_ON", "false").lower() == "true"


class Mem0Helper():

    def __init__(self, memory: Memory):
        self.memory = memory

    @classmethod
    def create(cls) -> 'Mem0Helper':
        config = {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "user": os.environ.get("PG_USER"),
                    "password": os.environ.get("PG_PASSWORD"),
                    "host": os.environ.get("PG_HOST"),
                    "port": os.environ.get("PG_PORT")
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": os.environ.get("EMBEDDER_MODEL"),
                    "api_key": os.environ.get("EMBEDDER_API_KEY"),
                    "openai_base_url": os.environ.get("EMBEDDER_BASE_URL"),
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MEMORY_MODEL if MEMORY_MODEL else "gpt-4o-mini"
                }
            },
        }
        if GRAPH_ON:
            config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": os.environ.get("NEO4J_URL"),
                    "username": os.environ.get("NEO4J_USERNAME"),
                    "password": os.environ.get("NEO4J_PASSWORD")
                }
            }
        memory = Memory.from_config(config_dict=config)
        return cls(memory)

    def try_get_memories(self, message: str, user_id: str = DEFAULT_USER_ID) -> str:
        relevant_memories = self.memory.search(
            query=message, user_id=user_id, limit=MEMORY_HISTORY_LIMIT)
        if len(relevant_memories) == 0:
            return None
        memories_str = f"\n记忆部分: \n  说明: 这些记忆用户可能已经遗忘, 提起时要有适当提示. 但是必须贴近主题, 有些记忆是多余的.\n  有关 {user_id} 的记忆: |\n"
        memories_str += "\n".join(
            f"  {entry['memory']}" for entry in relevant_memories["results"])
        return memories_str

    def add_memory(self, messages: List[Dict[str, str]], user_id: str = DEFAULT_USER_ID):
        messages = list(filter(lambda inp: inp['role'] != 'system', messages))
        self.memory.add(messages[-MEMORY_ADD_LIMIT:], user_id=user_id)
