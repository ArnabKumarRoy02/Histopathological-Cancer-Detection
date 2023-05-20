# Histopathological Cancer Detection using Deep Learning

This repository contains code and resources for histopathological cancer detection using deep learning models. The project aims to develop accurate and efficient models for classifying histopathological images into cancerous and non-cancerous categories.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Histopathological cancer detection plays a crucial role in diagnosing and treating cancer. This project focuses on leveraging deep learning techniques to automate cancer detection from histopathological images. By developing accurate models, we aim to assist pathologists in making faster and more reliable diagnoses.

## Installation

To use the code in this repository, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/ArnabKumarRoy02/Histopathological-Cancer-Detection.git
```

2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

Here's a brief overview of the contents of this repository: 
- `train.py`: This script is used to train a model on the dataset.
- `evaluate.py`: This script is used to evaluate a trained model on the test set.
- `model.pt`: This is a trained model.

To train a model, run the following command:

```shell
python train.py
```

To evaluate a trained model, run the following command:

```shell
python evaluate.py
```

## Dataset

The dataset used in this project is taken from a Kaggle Competition. The dataset contains 220025 histopathological images of lymph node sections. The images are labelled as 0 (non-cancerous) and 1 (cancerous). The dataset is split into train, validation and test sets. The train set contains 176020 images, the validation set contains 22003 images and the test set contains 22002 images. 

You can download the dataset from [here](https://www.kaggle.com/c/histopathologic-cancer-detection/data).

## Model Training

A trained model is provided in `model.pt`. However, feel free to experiment with different architectures or adapt the code to train your own models. Make sure to refer to the documentation and comments within the code for more details.

## Evaluation

We provide an evaluation script `evaluate.py` to assess the performance of the trained model on test data. The script generates relevant metrics and outputs the results. Make sure to provide the path to the saved model and the test data directory as command-line arguments.

## Contributing

Contributions to this project are welcome! If you have any suggestions, bug reports, or would like to contribute improvements, please submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
