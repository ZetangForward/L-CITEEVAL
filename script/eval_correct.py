import json
import argparse
import glob
from transformers import AutoTokenizer
from collections import Counter
from rouge import Rouge
from tqdm import tqdm
import numpy as np
import re
import string



def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def rouge_score(prediction, ground_truth, **kwargs):
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, help='data path to load the jsonl')
    parser.add_argument('--task', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--shot', type=str, default="1")
    parser.add_argument('--exp', type=str)
    args = parser.parse_args()

    print(args.task)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    refer_path = "data/L-CiteEval/{}/{}.json".format(args.exp, args.task)
    with open(refer_path, 'r') as f:
        refer = json.load(f)


    path = 'result/{}/{}/{}shot_{}.json'.format(args.exp, args.task, args.shot, args.model)

    samples = []

    if args.exp == 'l-citeeval':  ### l-citeeval
        length_lst = [8000, 16000, 24000, 32000, 40000, 48000]
    elif args.exp == 'l-citeeval-length':   ### l-citeeval-length
        length_lst = [8000, 16000, 32000]
    elif argx.exp == 'l-citeeval-hardness': ### l-citeeval-hardness
        length_lst = ['easy', 'medium', 'hard']

    samples = []
    len_samples = {i: [] for i in length_lst}

    with open(path, 'r') as f:
        temp = json.load(f)
        for i in range(len(temp)):
            samples.append(temp[i])

            if args.exp != 'l-citeeval-hardness':
                for length_id in range(len(length_lst)):
                    length = refer[i]['length']
                    if args.exp == 'l-citeeval' and args.task == 'multi_news' and length > 40000: 
                        len_samples[40000].append(temp[i])
                        break
                    lower_bound = length_lst[length_id-1] if length_id > 0 else 0
                    upper_bound = length_lst[length_id]
                    if low_bound < length < upper_bound:
                        len_samples[length_lst[length_id]].append(temp[i])
                        break
            elif args.exp == 'l-citeeval-hardness':
                len_samples[refer[i][level]].append(temp[i])

    
    empty_key = []
    for i in len_samples.keys():
        if len(len_samples[i]) == 0:
            empty_key.append(i)
    
    for i in empty_key:
        len_samples.pop(i)
    print(path)
    res = {}
    if args.task in ['narrativeqa', 'natural_questions', 'hotpotqa', '2wikimultihopqa', 'dialsim', 'locomo']:
        f1_score_total = 0
        precision_score_total = 0
        recall_score_total = 0
        res['f1'] = {}
        res['precision'] = {}
        res['recall'] = {}
        for leng in len_samples:
            f1_score_sample = 0
            precision_score_sample = 0
            recall_score_sample = 0

            for i in tqdm(len_samples[leng], total=len(len_samples[leng])):
                model_ans = i['generation'][0].strip()
                if '\n' in model_ans:
                    ind = model_ans.index('\n')
                    model_ans = model_ans[:ind]

                model_ans = remove_citations(model_ans)

                temp_f1_score = temp_precsion_score = temp_recall_score = 0
                if type(i['answer']) == str:
                    temp_f1_score, temp_precsion_score, temp_recall_score = qa_f1_score(model_ans, i['answer'])
                elif type(i['answer']) == int:
                    temp_f1_score, temp_precsion_score, temp_recall_score = qa_f1_score(model_ans, str(i['answer']))
                elif type(i['answer']) == list:
                    for j in i['answer']:
                        current_f1_score, current_precsion_score, current_recall_score = qa_f1_score(model_ans, j)

                        temp_f1_score = max(temp_f1_score, current_f1_score)
                        temp_precsion_score = max(temp_precsion_score, current_precsion_score)
                        temp_recall_score = max(temp_recall_score, current_recall_score)      
                else:
                    assert 0
                f1_score_sample += temp_f1_score
                precision_score_sample += temp_precsion_score
                recall_score_sample += temp_recall_score
            
            res['precision'][leng] = round(100 * precision_score_sample / len(len_samples[leng]),2)
            res['recall'][leng] = round(100 * recall_score_sample / len(len_samples[leng]),2)
            res['f1'][leng] = round(100 * f1_score_sample / len(len_samples[leng]),2)
            precision_score_total += precision_score_sample
            recall_score_total += recall_score_sample
            f1_score_total += f1_score_sample
        
        res['precision']['all'] =  round(100 * precision_score_total / len(samples),2)
        res['recall']['all'] =  round(100 * recall_score_total / len(samples),2)
        res['f1']['all'] = round(100 * f1_score_total / len(samples),2)


    elif args.task in ['qmsum', 'gov_report', 'multi_news']:
        rouge = Rouge()
        res['rouge-l'] = {}
        rouge_score_total = 0
        for leng in len_samples:
            rouge_score_sample = 0

            for i in len_samples[leng]:
                model_ans = i['generation'][0].strip()

                if 'summary' in model_ans[:100].lower():
                    try:
                        ind = model_ans.index(':')
                    except:
                        continue
                    model_ans = model_ans[ind+1:].strip()

                if '\n' in model_ans:
                    ind = model_ans.index('\n')
                    model_ans = model_ans[:ind]

                model_ans = remove_citations(model_ans)

                if model_ans == "":
                    score = 0
                    continue

                score = 0
                if type(i['answer']) == str:
                    score = rouge_score(model_ans, i['answer'])
                elif type(i['answer']) == list:
                    for j in i['answer']:
                        score = max(score, rouge_score(model_ans, j))
                else:
                    assert 0

                rouge_score_sample += score['rouge-l']['f']


            
            res['rouge-l'][leng] = round(100 * rouge_score_sample / len(len_samples[leng]),2)
            rouge_score_total += rouge_score_sample
            
        res['rouge-l']['all'] = round(100 * rouge_score_total / len(samples),2)



    elif args.task == 'counting_stars':

        res['acc'] = {}
        acc_score_total = 0

        for leng in len_samples:
            recall_lst = []
            precision_lst = []
            f1_lst = []
            num_lst = []
            correct_lst = []
            for i in len_samples[leng]:
                gold_ind_lst = []
                gold_ans_lst = []
                for j in range(len(i['docs'])):
                    if "The little penguin counted" in i['docs'][j]:
                        gold_ind_lst.append(j+1)
                        pattern = r'The little penguin counted (\d+) ★'
                        match = re.search(pattern, i['docs'][j])
                        gold_ans_lst.append(int(match.group(1)))

                assert len(gold_ans_lst) == len(i['answer'])
                model_ans = i['generation'][0].strip()
                try:
                    if "Llama3-ChatQA-2-70B" not in args.file:
                        ind1 = model_ans.index("{")
                        ind2 = model_ans.index('}')
                        model_ans = json.loads(model_ans[ind1:ind2+1])
                    else:
                        model_ans = json.loads('{' + model_ans + '}')
                except:
                    precision_lst.append(0)
                    recall_lst.append(0)
                    f1_lst.append(0)
                    num_lst.append(0)
                    correct_lst.append(0)
                    continue

                total_correct = cite_correct = 0

                if 'passage_id' not in model_ans:
                    precision_lst.append(0)
                    recall_lst.append(0)
                    f1_lst.append(0)
                    num_lst.append(0)
                else:

                    model_ans['passage_id'] = list(set(model_ans['passage_id']))

                    for idx, psg_id in enumerate(model_ans['passage_id']):

                        if psg_id in gold_ind_lst:
                            cite_correct += 1
                    
                    precision = cite_correct / len(model_ans['passage_id'])
                    recall = cite_correct / len(gold_ind_lst)
                    if precision + recall == 0:
                        f1 = 0
                    else:
                        f1 = 2 * precision * recall / (precision + recall)

                    precision_lst.append(precision * 100)
                    recall_lst.append(recall * 100)
                    f1_lst.append(f1 * 100)
                    num_lst.append(len(model_ans['passage_id']))

                model_ans['little_penguin'] = model_ans['little_penguin'][:len(gold_ans_lst)]
                for idx, ans in enumerate(model_ans['little_penguin']):

                    if ans in gold_ans_lst:
                        total_correct += 1

                correct_lst.append(total_correct/len(gold_ans_lst) * 100)

                res['acc'][leng] = round(np.mean(correct_lst), 2)
            acc_score_total += sum(correct_lst)

        res['acc']['all'] = round(acc_score_total / len(samples), 2)

        


    elif args.task == 'niah':
        rouge = Rouge()
        res['rouge-1'] = {}
        rouge_score_total = 0
        for leng in len_samples:
            rouge_score_sample = 0

            for i in len_samples[leng]:
                model_ans = i['generation'][0].strip()
                if '\n' in model_ans:
                    ind = model_ans.index('\n')
                    model_ans = model_ans[:ind]

                model_ans = remove_citations(model_ans)

                score = 0
                if type(i['answer']) == str:
                    score = rouge_score(model_ans, i['answer'])
                elif type(i['answer']) == list:
                    for j in i['answer']:
                        score = max(score, rouge_score(model_ans, j))
                else:
                    assert 0

                rouge_score_sample += score['rouge-1']['r']
            
            res['rouge-1'][leng] = round(100 * rouge_score_sample / len(len_samples[leng]),2)
            rouge_score_total += rouge_score_sample
            
        res['rouge-1']['all'] = round(100 * rouge_score_total / len(samples), 2)

    save_path = 'result/{}/{}/correct_score_{}shot_{}.json'.format(args.exp, args.task, args.shot, args.model)
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=2)