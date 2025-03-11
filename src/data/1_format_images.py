import pathlib
import subprocess
import os
import shutil

wd = pathlib.Path(__file__).parent.parent.parent
dst_path = wd / "data/interim/images"
shutil.rmtree(dst_path)
os.mkdir(dst_path)

src_path = wd / "data/raw/334003_253565_Alchemische-Objekte_002/images/train"
files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]

for f in files:
    src_ = wd / "data/raw/334003_253565_Alchemische-Objekte_002/images/train" / f
    shutil.copy(src_, dst_path)


subprocess.call("mogrify -format jpg *.png && rm *.png", shell=True, cwd=dst_path)