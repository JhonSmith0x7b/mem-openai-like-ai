from typing import List, Dict
from litserve.specs.openai import ChatMessage


def convert_openai_message_to_dict_message(messages: List[ChatMessage]) -> Dict[str, str]:
    converted_data = []
    for message in messages:
        converted_data.append({
            "role": message.role,
            "content": message.content if isinstance(message.content, str) else [c for c in message.content]
        })
    return converted_data