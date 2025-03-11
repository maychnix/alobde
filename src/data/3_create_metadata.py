import json
import pathlib
import os
from typing import List, Any
import copy

from jsonl import jsonl

wd = pathlib.Path(__file__).parent.parent.parent
files_path = "data/interim/json_labels"
path = (wd / files_path)

def create_metadata(path_files):
    # target_sequence
    target_sequence = {
        "ampullae": 0,
        "animal": 0,
        "cucurbitae": 0,
        "cucurbitae-ambix": 0,
        "cucurbitae-retorte": 0,
        "cucurbitae-rosenhut": 0,
        "furnace": 0,
        "human": 0,
        "mineral-metal": 0,
        "other-equipment": 0,
        "plant": 0,
        "ollae": 0
    }

    # get file names
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    metadata: List[Any] = []

    #
    for file in files:
        image_file = str(file).replace("json", "jpg")
        with open((path / file), 'r') as f:
            labels = json.load(f).get(file) # get -> to get only labels
        labels_out = copy.deepcopy(target_sequence)
        for key, val in labels.items():
            labels_out[key] = val
        line = {"image": image_file, "text": labels_out}
        metadata.append(line)

    return metadata


md = create_metadata(path)

jsonl.dump(md, (str(wd) + "/data/interim/metadata.jsonl"))


