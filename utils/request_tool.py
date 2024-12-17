import asyncio
from copy import deepcopy
import json
import os

from tqdm import tqdm
from utils.common_tool import read_jsonl
import google.generativeai as genai

class MMRequestor:
    def __init__(self,
                 model_type="gpt",
                 model_name="gemini-pro-vision",
                 api_key="YOUR_API_KEY",
                 enable_multi_turn=False,
                 request_proxy=None) -> None:
        
        self.model_type = model_type
        self.model_name = model_name
        self.enable_multi_turn = enable_multi_turn
        if model_type == "gpt":
            from openai import AsyncOpenAI
            if request_proxy is not None:
                client = AsyncOpenAI(api_key=api_key,
                                    base_url=request_proxy)
            else:
                client = AsyncOpenAI(api_key=api_key)
            self.requestor = client
            self.chat = []
        elif model_type == "gemini":
            genai.configure(api_key=api_key)
            self.requestor = genai.GenerativeModel(model_name)
            if enable_multi_turn:
                raise ValueError("Multiple turn dialog is not supported for Gemini")
        else:
            raise ValueError("Not Supported other model besides ['gpt', 'gemini']")
    
    async def request(self, prompts, **kargs):
        
        if self.model_type == "gpt":
            if isinstance(prompts, list):
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    response = await self.requestor.chat.completions.create(
                        model=self.model_name,
                        messages=self.chat,
                        **kargs
                        )
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": response.choices[0].message.content,
                        }]
                    })
            else:
                prompt = prompts
                self.chat.append({
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": prompt,
                        }],
                })
                response = await self.requestor.chat.completions.create(
                    model=self.model_name,
                    messages=self.chat,
                    **kargs
                    )
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": response.choices[0].message.content,
                    }]
                })
            res_str = deepcopy(self.chat)
            self.chat = []
            return res_str
        elif self.model_type == "gemini":
            if isinstance(prompts, list):
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    response = self.requestor.generate_content([x["content"][0]["text"] for x in self.chat])
                    
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": response.text
                        }]
                    })
            else:
                prompt = prompts
                self.chat.append({
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": prompt,
                        }],
                })
                response = self.requestor.generate_content([x["content"][0]["text"] for x in self.chat])
                
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": response.text
                    }]
                })
            res_str = deepcopy(self.chat)
            self.chat = []
            return res_str

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf8") as f:
        f.write(json_string + "\n")

async def producer(queue, dataset, save_path, bar, create_prompt):
    if os.path.exists(save_path):
        last_request = [x["index"] for x in read_jsonl(save_path)]
    else:
        last_request = []
    for i, data in enumerate(dataset.data):
        if data["index"] in last_request:
            bar.update(1)
            continue
        prompt = create_prompt(data)
        # DSP
        print("Loaded\t\t#", i)
        data.update({"index": str(i), "text": prompt})
        await queue.put(data)
    print("Dataset Loaded.")
    await queue.put(None)


async def consumer(queue,
                   save_path,
                   bar,
                   model_type,
                   model_name,
                   api_key,
                   enable_multi_turn,
                   request_proxy,
                   model_config
                   ):
    while True:
        item = await queue.get()
        if item is None:
            break
        text = item["text"]
        
        
        
        try:
            print("Requesting\t\t#", item["index"])
            requestor = MMRequestor(model_type=model_type,
                                    model_name=model_name,
                                    api_key=api_key,
                                    enable_multi_turn=enable_multi_turn,
                                    request_proxy=request_proxy)
            result = await requestor.request(
                prompts=text,
                **model_config
            )
            append_to_jsonl({"index": item["index"], "pred": result}, save_path)
        except Exception as e:
            raise(e)
        print("Saved\t\t#", item["index"])
        bar.update(1)
        queue.task_done()

async def request_LLM(
                model_type,
                model_name,
                api_key,
                enable_multi_turn,
                dataset=None,
                save_path = "",
                consumer_size=15,
                create_prompt_fn=None,
                request_proxy=None,
                model_config={}):
    queue = asyncio.Queue(maxsize=60)
    
    if dataset is None:
        return
    bar = tqdm(total=len(dataset.data), desc=f"Requesting {model_name}...")
    producer_task = asyncio.create_task(producer(queue, dataset, save_path, bar, create_prompt_fn))
    consumer_tasks = [asyncio.create_task(consumer(queue, save_path, bar, model_type, model_name, api_key, enable_multi_turn, request_proxy, model_config)) for _ in range(consumer_size)]
    
    await producer_task
    await queue.join()

    for task in consumer_tasks:
        task.cancel()