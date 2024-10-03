import os
import torch
import pdb
import json
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration,AutoModelForCausalLM,AutoConfig,LlamaForCausalLM, LlamaTokenizer, AutoModelForSeq2SeqLM
import transformers
import argparse
from torch.utils.data import Dataset, DataLoader
import yaml
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def make_doc_prompt(doc, doc_id, doc_prompt):

    if type(doc) == str:
        text = doc
    elif type(doc) == dict:
        if 'title' in doc:
            title = doc['title']
            text = doc['text'].strip('\n')
            if text[:len(title)+1] == title + '\n':
                text = text[len(title)+1:]
        else:
            text = doc['text'].strip('\n')

    return doc_prompt.replace("{P}", text).replace("{ID}", str(doc_id+1))



def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, test=False):

    if "{Q}" in prompt:
        prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    else:
        prompt = prompt.replace("{INST}", instruction)
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "")
        else:
            try:
                doc_list = item["docs"][:ndoc]
            except:
                import pdb;pdb.set_trace()
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])

            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer

    else:
        prompt = prompt.replace("{A}", "").rstrip() 

    return prompt

def get_instruction_template(task, prompt, sample, tokenizer):

    head_prompt = ""
    if task in ["dialsim"]:      
        head_prompt += make_demo(
            prompt['demos'][0], prompt=prompt["demo_prompt"], ndoc=1500, doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"].replace("<<<chatbox>>>", prompt['demo_role'])
        )
    else:
        head_prompt += make_demo(
            prompt['demos'][0], prompt=prompt["demo_prompt"], ndoc=1500, doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"]
        )
    head_prompt += prompt["demo_sep"]

    if task in ["dialsim"]:  
        head_prompt += make_demo(
            sample, prompt=prompt["demo_prompt"], ndoc=1500, doc_prompt=prompt["doc_prompt"],
            instruction=prompt["instruction"].replace("<<<chatbox>>>", sample['role']), test=True
        )
    else:
        head_prompt += make_demo(
            sample, prompt=prompt["demo_prompt"], ndoc=1500, doc_prompt=prompt["doc_prompt"],
            instruction=prompt["instruction"], test=True
        )

    try:
        head_prompt = tokenizer.apply_chat_template([{"role": "user", "content": head_prompt}], tokenize=False, add_generation_prompt=True)
    except:
        pass

    return head_prompt


def query_model(port, prompt, max_tokens=20, stop_token_ids=None, temperature=0, top_p=1):
    url = "http://localhost:{}/generate".format(port)
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop_token_ids": stop_token_ids
    }

    response = requests.post(url, headers=headers, json=data).json()['text'][0]

    return dict(text=response[len(prompt):])

def main(args):
    if args.task in ["narrtiveqa", "natural_questions", "hotpotqa", "2wikimultihopqa", "locomo", "dialsim"]:
        max_gen_len = 200
    elif args.task in ["niah", "counting_stars"]:
        max_gen_len = 128
    else:
        max_gen_len = 800

    file_path = args.eval_file


    save_path = "result/{}/{}/1shot_{}.json".format(args.exp, args.task, os.path.basename(args.model))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_gen_times = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model,padding_side='left',trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    if 'glm' not in args.model.lower():
        stop_token_ids = list(set(tokenizer.encode("\n", add_special_tokens=False) + config.eos_token_id if type(config.eos_token_id) == list else [config.eos_token_id]))
    else:
        stop_token_ids = list(set(config.eos_token_id))


    with open(args.prompt_file, 'r') as f:
        prompt = json.load(f)

    id2sample = {}
    
    with open(args.eval_file, 'r') as f:

        lst = json.load(f)
        print('save path:', save_path)

        content = []

        for i in lst:
            content.append(i)
            id2sample[str(i['id'])] = i
    

    for i in range(len(content)):
        content[i]['model_input'] = get_instruction_template(args.task, prompt, content[i], tokenizer)

    ports = [i for i in range(4100, 4100+args.num_port)]

    with ThreadPoolExecutor(max_workers=32) as executor:
        result = [executor.submit(query_model, ports[i % len(ports)], content[i]['model_input'], max_gen_len, stop_token_ids) for i in range(len(content))]
        for _ in tqdm(as_completed(result), total=len(result)): pass  # use tqdm to show progress

        responses = [r.result() for r in result]

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    f = open(save_path, 'w')

    res = []
    for i in range(len(responses)):
        dic = {}
        dic['id'] = i + 1
        if 'question' in content[i]:
            dic['question'] = content[i]['question']
        else:
            dic['question'] = None
        dic['answer'] = content[i]['answer']
        dic['generation'] = [responses[i]['text']]
        dic['model_input'] = content[i]['model_input']
        res.append(dic)

    json.dump(res, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--model", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--length", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--num_port", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    
    main(args)
