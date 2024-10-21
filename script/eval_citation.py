
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import argparse
from tqdm import tqdm
from nltk import sent_tokenize
import re
import logging
import copy
import numpy as np
import os
from transformers import pipeline


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def _run_nli_autoais(passage, claim):

    global pipe
    result = pipe([dict(text=passage, text_pair=claim)])[0]['label']
    inference = 1 if result == "entailment" else 0
    return inference


def compute_autoais(data, at_most_citations=None):

    def _format_document(doc):
        """Format document for AutoAIS."""

        if type(doc) == str:
            return "Passage: %s" % (doc)
        if type(doc) == dict:
            return "Title: %s\nPassage: %s" % (doc['title'], doc['text'])

    ais_scores = []
    ais_scores_prec = []
    ais_scores_f1 = []
    cite_num_lst = []

    for item in tqdm(data, total=len(data)):

        # Get sentences by using NLTK
        sents = sent_tokenize(item['generation'][0])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1

            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref]) # title+text

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)

            entail += joint_entail

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 


        citation_recall = entail / len(sents)
        citation_prec = entail_prec / total_citations if total_citations > 0 else 0
        if citation_recall + citation_prec == 0:
            citation_f1 = 0
        else:
            citation_f1 = 2 * citation_recall * citation_prec / (citation_recall + citation_prec)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))
        ais_scores_f1.append(citation_f1)
        cite_num_lst.append(total_citations)


    return ais_scores, ais_scores_prec, ais_scores_f1, cite_num_lst


def compute_niah_citation(data, at_most_citations=None):

    cite_recall = []
    cite_prec = []
    cite_f1 = []
    cite_num_lst = []

    for i in tqdm(range(len(data))):
        for j in range(len(data[i]['docs'])):
            if data[i]['answer'] in data[i]['docs'][j]:
                gold_ind = j
                break

        model_ans = data[i]['generation'][0].strip()

        if '\n' in model_ans:
            ind = model_ans.index('\n')
            model_ans = model_ans[:ind]

        ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", model_ans)][:at_most_citations]

        if gold_ind in ref:
            cite_recall.append(1)
            cite_prec.append(1/len(ref))
            f1 = (2 * 1 * 1/len(ref)) / (1 + 1/len(ref))
            cite_f1.append(f1)
        else:
            cite_recall.append(0)
            cite_prec.append(0)
            cite_f1.append(0)
        cite_num_lst.append(len(ref))

    return cite_recall, cite_prec, cite_f1, cite_num_lst

def compute_counting_stars_citation(data, at_most_citations=None):

    recall_lst = []
    precision_lst = []
    f1_lst = []
    num_lst = []

    for i in data:
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

            continue

        total_correct = cite_correct = 0

        if 'passage_id' not in model_ans:
            precision_lst.append(0)
            recall_lst.append(0)
            f1_lst.append(0)

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
            precision_lst.append(precision)
            recall_lst.append(recall)
            f1_lst.append(f1)
            num_lst.append(len(model_ans['passage_id']))

    return recall_lst, precision_lst, f1_lst, num_lst




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, help='data path to load the jsonl')
    parser.add_argument('--exp', type=str, help='data path to load the jsonl')
    parser.add_argument('--task', type=str)
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")

    args = parser.parse_args()

    pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli", device='cuda')

    with open(args.file, 'r') as f:
        lst = json.load(f)

    refer_path = "data/L-CiteEval/{}/{}.json".format(args.exp, args.task)
    with open(refer_path, 'r') as f:
        refer = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    if args.exp == 'l-citeeval':  ### l-citeeval
        length_lst = [8000, 16000, 24000, 32000, 40000, 48000]
    elif args.exp == 'l-citeeval-length':   ### l-citeeval-length
        length_lst = [8000, 16000, 32000]
    elif argx.exp == 'l-citeeval-hardness': ### l-citeeval-hardness
        length_lst = ['easy', 'medium', 'hard']

    filtered_lst = {i: [] for i in length_lst}

    assert len(lst) == len(refer)

    for i in tqdm(range(len(lst))):
        if args.exp != 'l-citeeval-hardness':
            for length_id in range(len(length_lst)):
                length = refer[i]['length']
                lst[i]['docs'] = refer[i]['docs']
                if args.exp == 'l-citeeval' and args.task == 'multi_news' and length > 40000: 
                    filtered_lst[40000].append(lst[i])
                    break
                lower_bound = length_lst[length_id-1] if length_id > 0 else 0
                upper_bound = length_lst[length_id]
                if lower_bound <= length < upper_bound:
                    filtered_lst[length_lst[length_id]].append(lst[i])
                    break
        elif args.exp == 'l-citeeval-hardness':
            lst[i]['docs'] = refer[i]['docs']
            filtered_lst[refer[i][level]].append(lst[i])


    cite_recall = []
    cite_prec = []
    cite_f1 = []
    total_cite_num_lst = []
    
    length_res = {}
    final_res = []
    for length in length_lst:
        length_res[length] = {}
        if len(filtered_lst[length]) <= 0:
            continue
        if args.task == 'niah':
            ais_scores, ais_scores_prec, ais_scores_f1, cite_num_lst = compute_niah_citation(filtered_lst[length], at_most_citations=args.at_most_citations)
        elif args.task == 'counting_stars':
            ais_scores, ais_scores_prec, ais_scores_f1, cite_num_lst = compute_counting_stars_citation(filtered_lst[length], at_most_citations=args.at_most_citations)
        else:
            ais_scores, ais_scores_prec, ais_scores_f1, cite_num_lst = compute_autoais(filtered_lst[length], at_most_citations=args.at_most_citations)

        cite_recall += ais_scores
        cite_prec += ais_scores_prec
        cite_f1 += ais_scores_f1
        total_cite_num_lst += cite_num_lst

        length_res[length]['cite_recall'] = ais_scores
        length_res[length]['cite_prec'] = ais_scores_prec
        length_res[length]['cite_f1'] = ais_scores_f1

        temp_res = {
            "length": length,
            "citation_rec": round(100 * np.mean(length_res[length]['cite_recall']), 2),
            "citation_prec": round(100 * np.mean(length_res[length]['cite_prec']), 2),
            "citation_f1": round(100 * np.mean(length_res[length]['cite_f1']), 2),
            "cite_num": np.mean(cite_num_lst)
        }

        temp_res['file_path'] = args.file
        temp_res['sample_num'] = len(filtered_lst[length])
        final_res.append(temp_res)

    temp_res = {
        "length": "total",
        "citation_rec": round(100 * np.mean(cite_recall), 2),
        "citation_prec": round(100 * np.mean(cite_prec), 2),
        "citation_f1": round(100 * np.mean(cite_f1), 2),
        "cite_num": np.mean(total_cite_num_lst)
    }
    temp_res['file_path'] = args.file
    temp_res['sample_num'] = len(cite_recall)
    final_res.append(temp_res)

    with open("result/{}/{}/{}.json".format(args.exp, args.task, "deberta_score_" + os.path.basename(args.file).split('.json')[0]), 'w') as f:
        f.write(json.dumps(final_res, indent=2) + '\n')
        f.flush()



        
