import dotenv
dotenv.load_dotenv(override=True)
import litserve as ls
from litserve.specs.openai import ChatMessage
import os
from openai import OpenAI
from typing import List, Dict
from memory import Mem0Helper
from util import utils
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting YuKiNo API...")

MODEL = os.environ.get("OPENAI_MODEL")

PER_DEVICE_WORKER = int(os.environ.get("LITSERVE_PER_DEVICE_WORKER"))


class PredictCallback(ls.Callback):

    def on_after_predict(self, lit_api: 'YuKiNoAPI'):
        if lit_api.inputs:
            lit_api.mem0Helper.add_memory(lit_api.inputs)
            lit_api.inputs = False


class YuKiNoAPI(ls.LitAPI):

    def setup(self, device):
        self.model = OpenAI()
        self.mem0Helper = Mem0Helper.create()
        self.inputs = False
        self.model_name = MODEL

    def decode_request(self, request):
        logging.info(f"Received request: {request}")
        self.model_name = request.model if request.model != None else MODEL
        return request.messages

    def predict(self, inputs: List[ChatMessage], context):
        inputs = utils.convert_openai_message_to_dict_message(inputs)
        inputs = self.inject_memory(inputs)
        self.inputs = inputs
        for chunck in self.model.chat.completions.create(
                model=self.model_name, messages=inputs, stream=True):
            yield chunck.choices[0].delta.content

    def inject_memory(self, inputs: List[Dict[str, str]]) -> List[ChatMessage]:
        user_last_message = inputs[-1]['content']
        memory = self.mem0Helper.try_get_memories(user_last_message)
        logging.info(f"Retrieved memory: {memory}")
        if memory == None:
            return inputs
        memory = f"\n**有关用户的记忆**:\n {memory}"
        for inp in inputs:
            if inp['role'] == "system":
                inp['content'] += memory
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
