import sys

import pandas as pd
import pathlib
import datasets
from pandas_image_methods import PILMethods
from PIL import Image
import huggingface_hub
import random

import tqdm
pd.api.extensions.register_series_accessor("pil")(PILMethods)

from jsonl import jsonl

# paths
wd = pathlib.Path(__file__).parent.parent.parent
f_loc = "data/interim/json_labels"
im_loc = "data/interim/images"
im_path = (wd /im_loc)

#create dataframe
def create_dataset(seed):
    df = pd.DataFrame(columns=["image","labels","idx","source"])

    metadata = jsonl.load(str(wd / "data/interim/metadata.jsonl"))
    i = 0

    for line in metadata:
        im_path = str(wd / im_loc / line["image"])
        im = Image.open(im_path)
        labels = line["text"]
        source = line["image"].replace(".jpg", "")
        df = pd.concat([pd.DataFrame([[im, labels, i, source]], columns=df.columns), df], ignore_index=True)
        i += 1

    # save as Dataset (issue with PIL and Dataset.from_pandas() function: https://github.com/huggingface/datasets/issues/4796)
    df_dict = df.to_dict(orient="list")
    dataset = datasets.Dataset.from_dict(df_dict)
    ds_train_devtest = dataset.train_test_split(test_size=0.3, seed=seed)
    ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=seed)


    ds_splits = datasets.DatasetDict({
        'train': ds_train_devtest['train'],
        'valid': ds_devtest['train'],
        'test': ds_devtest['test']
    })

    return ds_splits

def save_dataset(dataset):
    dataset.save_to_disk(str(wd /"data/processed/dataset.hf"))
    huggingface_hub.login(token=) #TODO insert personal token
    dataset.push_to_hub("") # TODO insert user/repository

### functions
def dict_add_to_val(dict, key, val):
    v = dict.get(key, 0)
    v += val
    dict[key] = v
    return dict

def get_label_freq(dataset_split):
    out = {}
    for sample in dataset_split:
        for key, value in sample.items():
            if value is None:
                value = 0
            out = dict_add_to_val(out, key, value)
    return out

def get_total_label_freq(dataset):
    out = {}
    splits = ["train","test","valid"]
    for split in splits:
        ds_split = dataset[split]["labels"]
        for sample in ds_split:
            for key, value in sample.items():
                if value is None:
                    value = 0
                out = dict_add_to_val(out, key, value)
    return out

def sum_dict_values(dic):
    sum = 0
    for value in dic.values():
        if value is None:
            value = 0
        sum += value
    return sum

def get_percentages(dic):
    percentages = {}
    sum = sum_dict_values(dic)
    for key, value in dic.items():
        pct = (value / sum) * 100
        percentages[key] = round(pct,2)
    return percentages

def get_max_diff(total_dist, split_dist):
    max_diff = 0
    for key, val in total_dist.items():
        val_spl = float(split_dist.get(key))
        diff = abs(float(val) - val_spl)
        if diff > max_diff:
                max_diff = diff
    return max_diff

def validate_seed_min_error(dataset, current_min_error):
    splits = ["train", "test", "valid"]
    total_dist = get_total_label_freq(dataset)
    total_dist_pst = get_percentages(total_dist)
    splits_max_error = 0
    for split in splits:
        ds_split = dataset[split]["labels"]
        ds_split_dist = get_label_freq(ds_split)
        ds_split_pst = get_percentages(ds_split_dist)
        error = get_max_diff(total_dist_pst, ds_split_pst)
        if error > splits_max_error:
            splits_max_error = error
    if current_min_error < splits_max_error:
        return current_min_error
    return splits_max_error

def find_best_valid_seed(iterations):
    seed_out = 42
    current_min_error = 100
    seeds = random.sample(list(range(iterations*10)), iterations)
    for i in tqdm.tqdm(range(iterations)):
        seed = seeds[i]
        ds = create_dataset(seed)
        error = validate_seed_min_error(ds, current_min_error)
        if error < current_min_error:
            seed_out = seed
            current_min_error = error
        if i % 1000 == 0:
            print(f"best current seed: {seed_out}, with error: {current_min_error}")
    return seed_out, current_min_error



###########################################
# TODO uncomment to look for seed with least error margin compared to whole dataset
#seed, error = find_best_valid_seed(100000)
#print(f"best seed: {seed}, with error: {error}")
# best seed: 9515095, with error: 1.8699999999999992
# best seed: 9000864, with error: 1.5399999999999991 -> 9/10
# best seed: 2253, with error: 1.32 -> 9/10
# best current seed: 335479, with error: 1.3200000000000003 -> 7/10

ds = create_dataset(2253)
save_dataset(ds)
