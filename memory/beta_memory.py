
from mem0 import Memory
from openai import OpenAI
import os
from typing import Dict, List


DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID")
MEMORY_HISTORY_LIMIT = int(os.environ.get("MEMORY_HISTORY_LIMIT"))
MEMORY_ADD_LIMIT = int(os.environ.get("MEMORY_ADD_LIMIT"))
MEMORY_MODEL = os.environ.get("MEMORY_MODEL")


class Mem0Helper():

    def __init__(self):
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
                    "model": "text-embedding-3-large"
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MEMORY_MODEL if MEMORY_MODEL else "gpt-4o-mini"
                }
            }
        }
        self.memory = Memory.from_config(config_dict=config)

    def try_get_memories(self, message: str, user_id: str = DEFAULT_USER_ID) -> str:
        relevant_memories = self.memory.search(
            query=message, user_id=user_id, limit=MEMORY_HISTORY_LIMIT)
        if len(relevant_memories) == 0:
            return None
        memories_str = "\n".join(
            f"- {entry['memory']}" for entry in relevant_memories["results"])
        return memories_str

    def add_memory(self, messages: List[Dict[str, str]], user_id: str = DEFAULT_USER_ID):
        messages = list(filter(lambda inp: inp['role'] != 'system', messages))
        self.memory.add(messages[-MEMORY_ADD_LIMIT:], user_id=user_id)
