from typing import List, Dict
from litserve.specs.openai import ChatMessage


def convert_openai_message_to_dict_message(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    converted_data = []
    for message in messages:
        converted_data.append({
            "role": message.role,
            "content": message.content if (isinstance(message.content, str) or message.content is None) else [c for c in message.content]
        })
    return converted_data


def convert_dict_message2chat_str(messages: List[Dict[str, str]], user_name: str, assistant_name: str) -> str:
    chat_str = ""
    for message in messages:
        chat_name = user_name if message['role'] == 'user' else assistant_name
        chat_str += f"{chat_name}: {message['content'].split('</think>')[-1]}\n"
    return chat_str
