import json

# path = "/nvme/zky/iclr2024/dataset/narrativeqa/narrativeqa240_processed_new_9_21.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/narrativeqa.json"

# path = "/nvme/zky/iclr2024/dataset/natural_questions/natural_questions_wo_blank_answer_9_21_chunk256.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/natural_questions.json"

# path = "/nvme/zky/iclr2024/dataset/2WikiMultihopQA/data_ids/dev_processed_sample40_len100_150_new_9_19_merged.jsonl"
# save_path = "/nvme/zky/iclr2025/dataset/main/2wikimultihopqa.json"

# path = "/nvme/zky/iclr2024/dataset/gov_report/gov_report_processed.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/gov_report.json"

# path = "/nvme/zky/iclr2024/dataset/multi_news/multi_news_sample_processed_100.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/multi_news.json"

# path = "/nvme/zky/iclr2024/dataset/QMSum/QMSum_sample_processed_new.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/qmsum.json"

# path = "/nvme/zky/iclr2024/dataset/locomo/locomo10_9_21_processed_new_chunk256.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/locomo.json"

path = "/nvme/zky/iclr2024/dataset/dialsim/dialsim_chunk256.json"
save_path = "/nvme/zky/iclr2025/dataset/main/dialsim.json"

# path = "/nvme/zky/iclr2024/dataset/niah/9_21/niah_processed_9_21.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/niah.json"

# path = "/nvme/zky/iclr2024/dataset/counting_stars/counting_stars_processed_9_21.json"
# save_path = "/nvme/zky/iclr2025/dataset/main/counting_stars.json"


with open(path, 'r') as f:
    lst = json.load(f)

final_res = []

for i in range(len(lst)):
    dic = {}
    dic['id'] = i + 1
    if 'dialsim' in path.lower():
        dic['role'] = lst[i]['role']
    dic['question'] = lst[i]['question']
    dic['answer'] = lst[i]['answer']
    dic['docs'] = lst[i]['docs']
    final_res.append(dic)

with open(save_path, 'w') as f:
    json.dump(final_res, f, indent=2)

