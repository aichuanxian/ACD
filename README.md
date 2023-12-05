# ACD
<p align="right"><i>Authors: Pitts</i></p> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the implementations of the system described in the paper ["XXX"]() on [XXX](http://noisy-text.github.io/2021/) at the [XXX](https://2021.emnlp.org) conference.

## Repository Structure
```
ACD
└── src
    ├── commons
    │   ├── globals.py
    │   └── utils.py
    ├── data # implementation of dataset class
    ├── modeling 
    │   ├── layers.py # implementation of neural layers
    │   ├── model.py # implementation of neural networks
    │   └── train.py # functions to build, train, and predict with a neural network
    ├── experiment.py # entire pipeline of experiments
    └── main.py # entire pipeline of our system

```

## Installation
We have updated the code to work with Python 3.10, Pytorch 2.0.1, and CUDA 11.7. If you use conda, you can set up the environment as follows:

```bash
conda create -n ACD python==3.10
conda activate ACD
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y transformers tqdm matplotlib pandas pylint
conda install -c conda-forge sklearn-contrib-lightning
```

Also, install the dependencies specified in the requirements.txt:
```
conda create --name <env> --file requirements.txt
```

## Data
In this repository, we provide some toy examples to play with the code. Due to the policy, we are not allowed to release the data. If you need, please email Shuguang Chen ([schen52@uh.edu](schen52@uh.edu)) and we will provide the following data:

```
XXX
```


## Running

We use config files to specify the details for every experiment (e.g., hyper-parameters, datasets, etc.). You can modify config files in the `configs` directory and run experiments with following command:

```
CUDA_VISIBLE_DEVICES=[gpu_id] python src/main.py --config /path/to/config
```

If you would like to run experiments with VisualBERT, please download the pretrained weights from [VisualBERT](https://github.com/uclanlp/visualbert/tree/master/visualbert) and replace `pretrained_weights` in the config file:

```json
    ...
    "model": {
        "name": "mner",
        "model_name_or_path": "bert-base-uncased",
        "pretrained_weights": "path/to/pretrained_weights",
        "do_lower_case": true,
        "output_attentions": false,
        "output_hidden_states": false
    },
    ...
```

## Citation
```
@inproceedings{chen-etal-2021-images,
    title = "Can images help recognize entities? A study of the role of images for Multimodal {NER}",
    author = "Chen, Shuguang  and
      Aguilar, Gustavo  and
      Neves, Leonardo  and
      Solorio, Thamar",
    booktitle = "Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wnut-1.11",
    pages = "87--96",
    abstract = "Multimodal named entity recognition (MNER) requires to bridge the gap between language understanding and visual context. While many multimodal neural techniques have been proposed to incorporate images into the MNER task, the model{'}s ability to leverage multimodal interactions remains poorly understood. In this work, we conduct in-depth analyses of existing multimodal fusion techniques from different perspectives and describe the scenarios where adding information from the image does not always boost performance. We also study the use of captions as a way to enrich the context for MNER. Experiments on three datasets from popular social platforms expose the bottleneck of existing multimodal models and the situations where using captions is beneficial.",
}
```

## Contact
Feel free to get in touch via email to schen52@uh.edu.
