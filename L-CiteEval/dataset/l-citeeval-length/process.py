import json

# path = "/nvme/zky/iclr2024/dataset/narrativeqa/ablation_length/narrativeqa_ablation_length.json"
# save_path = "/nvme/zky/iclr2025/dataset/l-citeeval-length/narrativeqa.json"

# path = "/nvme/zky/iclr2024/dataset/hotpotqa/ablation_length/dev_ablation_length.json"
# save_path = "/nvme/zky/iclr2025/dataset/l-citeeval-length/hotpotqa.json"

# path = "/nvme/zky/iclr2024/dataset/gov_report/ablation_length/gov_report_ablation_length.json"
# save_path = "/nvme/zky/iclr2025/dataset/l-citeeval-length/gov_report.json"

# path = "/nvme/zky/iclr2024/dataset/locomo/ablation_length/locomo10_ablation_length.json"
# save_path = "/nvme/zky/iclr2025/dataset/l-citeeval-length/locomo.json"

path = "/nvme/zky/iclr2024/dataset/counting_stars/ablation_length/counting_stars_ablation_length.json"
save_path = "/nvme/zky/iclr2025/dataset/l-citeeval-length/counting_stars.json"


with open(path, 'r') as f:
    lst = json.load(f)

final_res = []

print(len(lst))

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

