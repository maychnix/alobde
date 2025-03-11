# alobde
==============================


Task: Alchemic Object Detection (alobde) in historic book illustrations.


## Project Organization
------------

    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks used in inference and fine tuning. 
    │
    ├── reports            <- Generated analysis
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── metrics        <- Generated metrics to be used in reporting
    │   └── example_images <- Chosen images to be used in reporting
    │
    └── src                <- Source code for use in this project.
        ├── analysis       <- Scripts to analyse results and create output for reporting. 
        ├── data           <- Scripts to process raw data and generate dataset.
        └── visualization  <- Scripts to create plots.

------------
    
## How to reproduce results
1. Download data in YOLO format from supervisly and insert in data/raw.
2. Execute scripts in src/data in the order indicated my the file naming (first = 1_*, ...)
3. Execute notebooks with necessary ressources (inference < 11 GB VRAM, fine tuning ~ 45 GB VRAM)
4. Execute scripts in src/analysis in the order indicated my the file naming for the given analysis (first = 1_*, ...)

Data and reports folders are empty by design and will be filled by executing the scripts - given you have access to the data.

--------

<p><small>Project structure inspired by the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
