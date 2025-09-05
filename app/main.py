import dotenv
dotenv.load_dotenv(override=True)
import litserve as ls
from litserve.specs.openai import ChatMessage
import os
from openai import OpenAI
from typing import List, Dict
from memory import Mem0Helper
from util import utils
import copy
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting YuKiNo API...")

MODEL = os.environ.get("OPENAI_MODEL", "deepseek-chat")
TEMPERATURE = 0.7
TOP_P = 1.0
PRESENCE_PENALTY = 0.0

PER_DEVICE_WORKER = int(os.environ.get("LITSERVE_PER_DEVICE_WORKER", 2))


class PredictCallback(ls.Callback):

    def on_after_predict(self, lit_api: 'YuKiNoAPI'):
        if lit_api.inputs:
            temp = copy.deepcopy(lit_api.inputs)
            lit_api.inputs = None
            lit_api.mem0Helper.add_memory(temp)


class YuKiNoAPI(ls.LitAPI):

    def setup(self, device):
        self.model = OpenAI()
        self.mem0Helper = Mem0Helper.create()
        self.inputs = None
        self.model_name = MODEL
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.presence_penalty = PRESENCE_PENALTY

    def decode_request(self, request):
        logging.info(f"Received request: {request}")
        self.model_name = request.model if request.model != None else MODEL
        self.temperature = request.temperature if request.temperature != None else TEMPERATURE
        self.top_p = request.top_p if request.top_p != None else TOP_P
        self.presence_penalty = request.presence_penalty if request.presence_penalty != None else PRESENCE_PENALTY
        return request.messages

    def predict(self, inputs: List[ChatMessage], context):
        converted_inputs = utils.convert_openai_message_to_dict_message(inputs)
        converted_inputs = self.inject_memory(converted_inputs)
        self.inputs = converted_inputs
        try:
            for chunck in self.model.chat.completions.create(
                    model=self.model_name, messages=converted_inputs, stream=True, # type: ignore
                    temperature=self.temperature, top_p=self.top_p, presence_penalty=self.presence_penalty):
                yield chunck.choices[0].delta.content
        except Exception as e:
            yield f"ERROR {e}"

    def inject_memory(self, inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        user_last_message = inputs[-1]['content']
        memory = self.mem0Helper.try_get_memories(user_last_message)
        if memory == None:
            return inputs
        for inp in inputs:
            if inp['role'] == "system":
                inp['content'] += memory
                logging.info(f"Injected memory into system prompt. \n{inp}")
                return inputs
        inputs.insert(
            0, {'role': "system", 'content': f"你是个非常有用的AI, 你需要根据有关用户的记忆进行适当的回答 {memory}"})
        return inputs


if __name__ == "__main__":
    server = ls.LitServer(
        YuKiNoAPI(spec=ls.OpenAISpec(), enable_async=False, stream=True),
        callbacks=[PredictCallback()], workers_per_device=PER_DEVICE_WORKER
    )
    server.run(host="0.0.0.0", port=8086, generate_client_file=False)
