import pickle
from collections import Counter
import json
import pathlib
import os
import shutil

wd = pathlib.Path(__file__).parent.parent.parent


with open(os.path.join(wd, "data/interim/labels_file_total.pkl"), "rb") as f:
    l_f = pickle.load(f)

conversion_dict = ({
    0:"ampullae", #
    1:"animal", #
    2:"cucurbitae", #
    3:"cucurbitae-ambix", #
    4:"ollae", #
    5:"cucurbitae-retorte",#
    6:"cucurbitae-rosenhut", #
    7:"furnace", #
    8:"human", #
    9:"mineral-metal", #
    10:"other-equipment", #
    11:"plant" #
})

src = wd / "data/interim/json_labels"
shutil.rmtree(wd /"data/interim/json_labels")
os.mkdir(src)

for key, value in l_f.items():
    val = [conversion_dict.get(v, v) for v in value]
    count = Counter(val)
    name = str(key).replace("txt", "json")
    out = {}
    out[name] = count
    with open((src / name), "w") as f:
        json.dump(out, f)
