
from mem0 import Memory
from openai import OpenAI
import os


DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID")


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
            }
        }
        self.memory = Memory.from_config(config_dict=config)

    def try_get_memories(self, message: str, user_id: str = DEFAULT_USER_ID) -> str:
        relevant_memories = self.memory.search(
            query=message, user_id=user_id, limit=3)
        if len(relevant_memories) == 0:
            return None
        memories_str = "\n".join(
            f"- {entry['memory']}" for entry in relevant_memories["results"])
        return memories_str

    def add_memory(self, messages: list, user_id: str = DEFAULT_USER_ID):
        self.memory.add(messages, user_id=user_id)
