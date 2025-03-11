import pickle
import pathlib
import pandas
import pandas as pd

wd = pathlib.Path(__file__).parent.parent.parent
file = "data/processed/class_perf_qwen2_5.pkl"
file_path = (wd / file)

# get file names
with open(file_path, "rb") as fb:
        class_perf = pickle.load(fb)


####### metrics
## precison = tp/(tp+fp)

def calc_precision(dict, round_dec = 4):
    tp = dict.get("TP")
    fp = dict.get("FP")
    if (tp + fp) == 0:
        out = 0
    else:
        out = tp/(tp+fp)
    return round(out, round_dec)


## recall = tp/(tp+fn)
def calc_recall(dict, round_dec = 4):
    tp = dict.get("TP")
    fn = dict.get("FN")
    if (tp+fn) == 0:
        out = 0
    else:
        out = tp/(tp+fn)
    return round(out, round_dec)

## f1 = 2 * prec * rec/(prec+rec)
def calc_f1(dict, round_dec = 2):
    rec = calc_recall(dict)
    prec = calc_precision(dict)
    if (prec + rec) == 0:
        out = 0
    else:
        out = 2 * ((prec * rec) / (prec + rec))
    return round(out, round_dec)

#####
data = {}
for k, v in class_perf.items():
    r = calc_recall(v)
    p = calc_precision(v)
    f1 = calc_f1(v)
    data[k]=[r,p,f1]


## store
df = pd.DataFrame(data)
df.index = ["Recall", "Precision", "F1-Score"]
file_name = (wd / "reports/metrics/metrics_qwen2_5_untrained.csv")
print(df)
df.to_csv(file_name)