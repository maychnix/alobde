import pickle
import pathlib
import os

wd = pathlib.Path(__file__).parent.parent.parent
files_path = "data/processed/results/raw_qwen_2_5_test_dataset"
path = (wd / files_path)

# get file names
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

results = []
for f in files:
    file_path = path / f
    with open(file_path, "rb") as fb:
        r = pickle.load(fb)
    results.append(r)

results = [x for xs in results for x in xs]


## get dataset
import huggingface_hub
import datasets

huggingface_hub.login(token="") # TODO insert token
test_dataset = datasets.load_dataset("", split="test") # TODO insert repository


# compare output to groundtruth
assert len(test_dataset) == len(results)

class_perf = {"ampullae" : {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "animal": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "cucurbitae": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "cucurbitae-ambix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "cucurbitae-retorte": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "cucurbitae-rosenhut": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "furnace": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "human": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "mineral-metal": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "other-equipment": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "plant": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
              "ollae": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
              }

# helper functions
def update_dict(dict_to_update, key, value_to_add):
    t = dict_to_update.get(key)
    t += value_to_add
    dict_to_update[key] = t
    return dict_to_update



for i in range(len(test_dataset)):
    output = results[i]
    sample = test_dataset[i]

    # verify sample and response ids
    assert sample["idx"] == output["idx"]

    ground_truth = sample["labels"]
    model_answer = eval(output["output"].replace("json", "").replace("```",""))

    for k, v in ground_truth.items():
        v_m = model_answer.get(k)

        if v_m is None:
            v_m = 0

        if (v == 0) & (v_m == 0):
            t = class_perf.get(k)
            t = update_dict(t, "TN", 1)
            class_perf[k] = t
        else:
            if v == v_m:
                t = class_perf.get(k)
                t = update_dict(t, "TP", 1)
                class_perf[k] = t
            if v > v_m:
                t = class_perf.get(k)
                t = update_dict(t, "FN", (1 * (v-v_m)))
                t = update_dict(t, "TP", (1 * v_m))
                class_perf[k] = t
            if v_m > v:
                t = class_perf.get(k)
                t = update_dict(t, "FP", (1 * (v_m - v)))
                t = update_dict(t, "TP", (1 * v))
                class_perf[k] = t




with open((wd / "data/processed/class_perf_qwen2_5.pkl"), "wb") as file:
    pickle.dump(class_perf, file)
