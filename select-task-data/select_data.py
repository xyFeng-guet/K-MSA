import pickle
import numpy as np
import argparse
import pandas as pd


def options():
    parser = argparse.ArgumentParser(description="Select samples from a dataset")
    parser.add_argument("--dataset_name", type=str, default="SIMS", required=False, help="Name of the dataset")
    parser.add_argument("--num_samples", type=int, default="100", required=False, help="Number of samples to select")
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


def select_samples(data, num_samples):
    samples_list = []

    for i in range(len(data)):
        sample = data[i]
        if sample["label"] >= 0 and sample["label_v"] < 0 or sample["label"] <= 0 and sample["label_v"] > 0 or sample["label"] != 0 and sample["label_v"] == 0:     # 选择某一模态情感与多模态情感不符合的样例
            sample["error"] = abs(sample["label"] - sample["label_v"])
            samples_list.append(sample)
        if len(samples_list) >= num_samples:
            break

    # 将选中的样例保存为csv文件
    selected_samples = pd.DataFrame(samples_list)
    selected_samples = selected_samples.sort_values(by="error", ascending=False)    # 按照error列的大小排序
    selected_samples.to_csv("./select-task-data/selected_samples_v.csv", index=False)
    return data


if __name__ == '__main__':
    args = options()
    data = get_data(args.dataset_name)
    selected_data = select_samples(data, args.num_samples)
    print(f"Selected {len(selected_data)} samples from {args.dataset_name}")
