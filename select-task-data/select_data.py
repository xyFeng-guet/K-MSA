import pickle
import numpy as np
import argparse
import pandas as pd


def options():
    parser = argparse.ArgumentParser(description="Select samples from a dataset")
    parser.add_argument("--dataset_name", type=str, default="SIMS", required=False, help="Name of the dataset")
    parser.add_argument("--num_samples", type=int, default="10000", required=False, help="Number of samples to select")
    parser.add_argument("--match_modality", type=str, default="consistency", required=False, choices=["t", "v", "a", "multimodal", "consistency"], help="Modality to match with the label.")
    args = parser.parse_args()
    return args


def get_data(dataset_name):
    dataset_path = f"/opt/data/private/Project/Datasets/MSA_Datasets/{dataset_name}/Processed/unaligned_39.pkl"
    # with open(dataset_path, 'rb') as f:
    #     data = pickle.load(f)

    label_path = f"/opt/data/private/Project/Datasets/MSA_Datasets/{dataset_name}/label.csv"
    labels = pd.read_csv(label_path)

    data_samples = []
    video_id = labels['video_id']
    clip_id = labels['clip_id']
    text = labels['text']
    label = labels['label']
    label_t = labels['label_T']
    label_v = labels['label_V']
    label_a = labels['label_A']
    for i in range(len(label)):
        data_samples.append({
            'video_id': video_id[i],
            'clip_id': clip_id[i],
            'text': text[i],
            'label': label[i],
            'label_t': label_t[i],
            'label_v': label_v[i],
            'label_a': label_a[i]
        })

    return data_samples


def select_samples(data, num_samples, match_modality):
    samples_list = []

    for i in range(len(data)):
        sample = data[i]
        if sample["label"] >= 0 and sample[f"label_{match_modality}"] < 0 or sample["label"] <= 0 and sample[f"label_{match_modality}"] > 0 or sample["label"] != 0 and sample[f"label_{match_modality}"] == 0:     # 选择某一模态情感与多模态情感不符合的样例
            sample["error"] = abs(sample["label"] - sample[f"label_{match_modality}"])
            samples_list.append(sample)
        if len(samples_list) >= num_samples:
            break

    # 将选中的样例保存为csv文件
    selected_samples = pd.DataFrame(samples_list)
    selected_samples = selected_samples.sort_values(by="error", ascending=False)    # 按照error列的大小排序
    selected_samples.to_csv(f"./select-task-data/selected_samples_{match_modality}.csv", index=False)
    return data


# 选取交集（多个模态不符合最终情感）
def select_samples_intersection(num_samples, match_modality):
    if match_modality != "multimodal":
        raise ValueError("match_modality must be multimodal")

    samples_list_t = pd.read_csv("./select-task-data/selected_samples_t.csv").to_dict(orient="records")
    samples_list_v = pd.read_csv("./select-task-data/selected_samples_v.csv").to_dict(orient="records")
    samples_list_a = pd.read_csv("./select-task-data/selected_samples_a.csv").to_dict(orient="records")

    # 求字典列表的交集
    samples_list_tv = [sample for sample in samples_list_t if sample in samples_list_v]
    samples_list_ta = [sample for sample in samples_list_t if sample in samples_list_a]
    samples_list_va = [sample for sample in samples_list_v if sample in samples_list_a]

    # 保存交集样例, 并按照error列升序排序
    selected_samples_tv = pd.DataFrame(samples_list_tv)
    selected_samples_tv = selected_samples_tv.sort_values(by="error", ascending=False)
    selected_samples_tv.to_csv("./select-task-data/selected_samples_tv.csv", index=False)

    selected_samples_ta = pd.DataFrame(samples_list_ta)
    selected_samples_ta = selected_samples_ta.sort_values(by="error", ascending=False)
    selected_samples_ta.to_csv("./select-task-data/selected_samples_ta.csv", index=False)

    selected_samples_va = pd.DataFrame(samples_list_va)
    selected_samples_va = selected_samples_va.sort_values(by="error", ascending=False)
    selected_samples_va.to_csv("./select-task-data/selected_samples_va.csv", index=False)

    return [samples_list_tv, samples_list_ta, samples_list_va]


def select_samples_consistency(data, num_samples, match_modality):
    samples_list = {
        "t": [],
        "v": [],
        "a": []
    }

    for i in range(len(data)):
        sample = data[i]
        max_value = np.min([abs(sample["label"] - sample["label_t"]), abs(sample["label"] - sample["label_v"]), abs(sample["label"] - sample["label_a"])])

        if max_value == abs(sample["label"] - sample["label_t"]):
            sample["error"] = abs(sample["label"] - sample["label_t"])
            samples_list["t"].append(sample)

        if max_value == abs(sample["label"] - sample["label_v"]):
            sample["error"] = abs(sample["label"] - sample["label_v"])
            samples_list["v"].append(sample)

        if max_value == abs(sample["label"] - sample["label_a"]):
            sample["error"] = abs(sample["label"] - sample["label_a"])
            samples_list["a"].append(sample)

        if len(samples_list) >= num_samples:
            break

    # 将选中的样例保存为csv文件
    selected_samples_t = pd.DataFrame(samples_list["t"])
    selected_samples_t = selected_samples_t.sort_values(by="error", ascending=True)
    selected_samples_t.to_csv("./select-task-data/selected_samples_consistency_t.csv", index=False)

    selected_samples_v = pd.DataFrame(samples_list["v"])
    selected_samples_v = selected_samples_v.sort_values(by="error", ascending=True)
    selected_samples_v.to_csv("./select-task-data/selected_samples_consistency_v.csv", index=False)

    selected_samples_a = pd.DataFrame(samples_list["a"])
    selected_samples_a = selected_samples_a.sort_values(by="error", ascending=True)
    selected_samples_a.to_csv("./select-task-data/selected_samples_consistency_a.csv", index=False)

    return samples_list


if __name__ == '__main__':
    args = options()
    data = get_data(args.dataset_name)

    if args.match_modality in ["t", "v", "a"]:
        selected_data = select_samples(data, args.num_samples, args.match_modality)
    elif args.match_modality == "multimodal":
        # select multimodal samples
        selected_data = select_samples_intersection(args.num_samples, args.match_modality)
    else:
        # select the samples of most consistency between the modalities
        selected_data = select_samples_consistency(data, args.num_samples, args.match_modality)

    print(f"Selected {len(selected_data)} samples from {args.dataset_name}")

'''
SIMS数据集总共2281个样例

T 情感与多模态情感不符合的样例有: 705个   30.91%
V 情感与多模态情感不符合的样例有: 499个   21.88%
A 情感与多模态情感不符合的样例有: 504个   22.10%

T+V 情感与多模态情感不符合的样例有: 58个  2.54%
T+A 情感与多模态情感不符合的样例有: 106个 4.65%
V+A 情感与多模态情感不符合的样例有: 91个  3.99%

单模态情感与多模态情感最一致
T: 1058个   46.38%
V: 1344个   58.92%
A: 1292个   56.64%
'''
