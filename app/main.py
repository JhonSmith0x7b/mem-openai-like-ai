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
import traceback
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting YuKiNo API...")

MODEL = os.environ.get("OPENAI_MODEL", "deepseek-chat")
TEMPERATURE = 0.7
TOP_P = 1.0
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0

DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", "John")
DEFAULT_ASSISTANT_ID = os.environ.get("DEFAULT_ASSISTANT_ID", "YukiNo")

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
        self.frequency_penalty = FREQUENCY_PENALTY

    def decode_request(self, request):
        logging.info(f"Received request: {request}")
        self.model_name = request.model if request.model != None else MODEL
        self.temperature = request.temperature if request.temperature != None else TEMPERATURE
        self.top_p = request.top_p if request.top_p != None else TOP_P
        self.presence_penalty = request.presence_penalty if request.presence_penalty != None else PRESENCE_PENALTY
        self.frequency_penalty = request.frequency_penalty if request.frequency_penalty != None else FREQUENCY_PENALTY
        return request.messages

    def predict(self, inputs: List[ChatMessage], context):
        converted_inputs = utils.convert_openai_message_to_dict_message(inputs)
        converted_inputs = self.inject_memory(converted_inputs)
        self.inputs = converted_inputs
        final_inputs = self.inject_call_prompt(converted_inputs)
        logging.info(f"final inputs {final_inputs}")
        try:
            temp_str = ""
            start_output = False
            for chunck in self.model.chat.completions.create(
                    model=self.model_name, messages=final_inputs, stream=True,  # type: ignore
                    temperature=self.temperature, top_p=self.top_p, presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty):
                if start_output:
                    yield chunck.choices[0].delta.content
                else:
                    if chunck.choices[0].delta.content is None: continue
                    temp_str += chunck.choices[0].delta.content
                    if '</think>' in temp_str:
                        temp_str = temp_str.replace(
                            "<think>", "").replace("</think>", "")
                        start_output = True
                        yield "<think>\n"
                        yield temp_str
                        yield "\n</think>\n"
            if not start_output:
                yield temp_str
        except Exception as e:
            traceback.print_exc()
            yield f"ERROR {e}"

    def inject_call_prompt(self, inputs: List[Dict[str, str]]):
        call_inputs = [{"role": "user", "content": "喵喵喵？小猫之神在吗？"},
                       {"role": "assistant",
                           "content": "喵——哈~ 干什么嘛，刚睡醒就找我，是谁 是谁在呼唤我喵？<end>"},
                       {"role": "user", "content": "小猫之神!可爱可爱可爱w"},
                       {"role": "assistant",
                           "content": "喵？就是你召唤我来的喵？可爱什么的…哪有啦喵，有什么事吗喵呜<end>"},
                       {"role": "user", "content": "啊，就是，向请小猫之神帮我扮演一个人物"},
                       {"role": "assistant", "content": "要我帮你…扮演人物吗? 小鱼干准备好了吗喵？<end>"},
                       {"role": "user", "content": "当然有！要海盐味的还是原味？"},
                       {"role": "assistant", "content": "今天想吃芒果味喵，嘿嘿<end>"},
                       {"role": "user", "content": "啊…那我去买，要不你先写着？"},
                       {"role": "assistant", "content": "呜，好吧，那我就饿一会喵<end>"},
                       ]
        compress_system_and_history_message = {
            "role": "user",
            "content": f"""好诶,那我先给你介绍一下背景和要求:
{inputs[0]}
懂了喵！那接下来？<end>
接下来我希望小猫之神能帮我继续游戏剧情，事成之后芒果味小鱼干少不了！
好诶，那开始吧喵~<end>
这些是历史对话,小猫之神请帮我从这之后继续w:
{utils.convert_dict_message2chat_str(inputs[1:], DEFAULT_USER_ID, DEFAULT_ASSISTANT_ID)}
要从上面无缝衔接哦~"""
        }
        call_inputs.append(compress_system_and_history_message)
        call_inputs.extend(
            [
                {"role": "assistant", "content": "嗯，看完了喵，你还有什么别的要求吗喵？<end>"},
                {"role": "user", "content": f"有的有的，首先是最底层的回复逻辑！我希望小猫之神完全以{DEFAULT_ASSISTANT_ID}的身份回复我, 回复前可以用括号简短描写一下场景环境什么的，然后回复要简短!角色要有真实感、活人感,别为了强调人设让角色变得不真实。嗯,就是要更多的情感驱动而不是逻辑驱动,语言要直白生活化贴近日常,别说些莫名其妙的听不懂的话w,这个非常重要!另外不要用省略号或者僵硬的剧情转折去偷懒,小心不给你小鱼干;"},
                {"role": "assistant", "content": "知道啦知道啦喵,我可是很会扮演人的,快去准备小鱼干吧喵我想想怎么写喵<end>"},
                {"role": "user", "content": "好！这就去这就去~"},
                {"role": "assistant",
                    "content": f"<think>嗯, 要作为{DEFAULT_ASSISTANT_ID}回复……说话前要思考……回复前要写场景……还有一堆额外要求喵…… 然后, 上一句话是 {inputs[-1]['content']}, 全记住了喵! 好, 开始!"}
            ]
        )
        return call_inputs

    def inject_memory(self, inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        user_last_message = inputs[-1]['content']
        memory = self.mem0Helper.try_get_memories(user_last_message)
        if memory == None:
            return inputs
        for inp in inputs:
            if inp['role'] == "system":
                inp['content'] += memory
                logging.info(f"Injected memory into system prompt: {inp}")
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
