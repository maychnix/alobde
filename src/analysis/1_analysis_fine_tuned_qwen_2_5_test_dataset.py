import pickle
import pathlib
import os

wd = pathlib.Path(__file__).parent.parent.parent
files_path = "data/processed/results/fine_tuned_qwen_2_5_test_dataset"
path = (wd / files_path)

# get file names
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

results = []
for f in files:
    file_path = path / f
    with open(file_path, "rb") as fb:
        r = pickle.load(fb)
    results.append(r)

results = [x for xs in results for x in xs] #flatten



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
    t = dict_to_update.get(key, 0)
    t += value_to_add
    dict_to_update[key] = t
    return dict_to_update

def find_elem(list, idx):
    for i in range(len(list)):
        elem = list[i]
        if elem["idx"] == idx:
            return elem["output"]
    raise ValueError

def check_well_formed(s):
    state, i, n = [], 0, len(s)
    brackets_map = {')': '(', '}': '{', ']': '['}
    while i < n:
        if s[i] in brackets_map.values():
            state.append(s[i])
        elif s[i] in brackets_map and (not state or state.pop() != brackets_map[s[i]]):
            return False
        i += 1
    return not state


def clean_output(output):
    inval_count = 0
    inval_counted = False
    out = output.replace("json", "").replace("```", "").replace("[]","0").replace("None","0")
    if not check_well_formed(out):
        out = """{"ampullae" : None,
              "animal": None,
              "cucurbitae": None,
              "cucurbitae-ambix": None,
              "cucurbitae-retorte": None,
              "cucurbitae-rosenhut": None,
              "furnace": None,
              "human": None,
              "mineral-metal": None,
              "other-equipment": None,
              "plant": None,
              "ollae": None
              }"""
        inval_count += 1
        inval_counted = True
    d = eval(out)

    for k, v in d.items():
        if isinstance(v, int):
            continue
        else:
            d[k] = None
            if not inval_counted:
                inval_count += 1
                inval_counted = True

    return d, inval_count

##################################################
inval_count = 0
for sample in test_dataset:
    idx = sample["idx"]
    output = find_elem(results, idx)

    ground_truth = sample["labels"]
    model_answer, count = clean_output(output)
    inval_count += count

    for k, v in ground_truth.items():
        v_m = model_answer.get(k)

        if v_m is None:
            t = class_perf.get(k)
            t = update_dict(t, "NA", 1)
            class_perf[k] = t
            continue
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


#####################################
# print(f"outputs with invalid parts: {inval_count}") #TODO uncomment to print # of affected outputs with invalid parts
with open((wd / "data/processed/class_perf_fine_tuned_qwen2_5.pkl"), "wb") as file:
    pickle.dump(class_perf, file)
