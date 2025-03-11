import pickle
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

wd = pathlib.Path(__file__).parent.parent.parent
file = "data/processed/class_perf_fine_tuned_qwen2_5.pkl"
file_path = (wd / file)

# get file names
with open(file_path, "rb") as fb:
        class_perf = pickle.load(fb)

## helper functions
def convert_to_matrix(dic):
    out = [[0,0],[0,0]]
    na_count = 0
    out[1][1] = dic.get("TP")
    out[1][0] = dic.get("FP")
    out[0][1] = dic.get("FN")
    out[0][0] = dic.get("TN")
    na_count = dic.get("NA")

    return out, na_count

### plots

fig, axs = plt.subplots(4, 3)
i = 0
j = 0
for k,v in class_perf.items():
    cm, na_count = convert_to_matrix(v)
    sns.heatmap(cm,
                annot=True,
                cbar=False,
                ax=axs[i,j],
                cmap=sns.cubehelix_palette(as_cmap=True))
    axs[i,j].set_title(k)
    axs[i, j].set_ylabel(("#NA:" + str(na_count)))
    if j >= 2:
        i += 1
        j = 0
    else:
        j += 1

fig.supxlabel("True Labels",
              size = 18)
fig.supylabel("Predicted Labels",
              size = 18)
fig.tight_layout()

plt.show()
