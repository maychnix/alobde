import pickle
import matplotlib.pyplot as plt
import pathlib

import huggingface_hub
import datasets

wd = str(pathlib.Path(__file__).parent.parent.parent)

huggingface_hub.login(token="") # TODO insert token
test_dataset = datasets.load_dataset("", split="test") # TODO insert repository

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

def get_barplot(dataset, name, total = False):
    if total:
       d = get_total_label_freq(dataset)
    else:
        d = get_label_freq(dataset)
    names, counts = zip(*d.items())
    graph = plt.bar(names, counts)

    percentages = get_percentages(d)

    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height * 1.01,
                 str(percentages.get(names[i])) + '%',
                 ha='center',
                 weight='bold')
        i += 1

    plt.xticks(rotation=45)
    plt.savefig(wd + '/reports/figures/' + name + '_label_distribution.png')
    plt.clf()

##############
train_ds = ds["train"]["labels"]
test_ds = ds["test"]["labels"]
val_ds = ds["valid"]["labels"]

get_barplot(ds, "total", True)
get_barplot(train_ds, "train")
get_barplot(test_ds, "test")
get_barplot(val_ds, "val")
