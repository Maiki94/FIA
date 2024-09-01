# FIA

Assigment for the FIA, the goal is to build two models for Fraud dectection.

## ⚙️ Installation & Setup

To install it you just need to run the following command in an environment with Python
3.11 or higher:

> `poetry install --all-extras`

In case you want to update the dependencies once installed the first version, you just
need to:

> `poetry update`

And the `poetry.lock` file will be updated too, if applicable.

## 📁 Project Structure

```
📂 FIA
├── 📂 data                    - data files
├── 📂 fia                     - the main package
|   ├── 📂 nn                  - code for neural network
|   |   ├── 🐍 classifier.py   - neural network definition
|   |   ├── 🐍 data.py         - data module
|   |   ├── 🐍 metrics.py      - torchmtrics
|   |   ├── 🐍 model.py        - ligthtining module
|   |   └── 🐍 train.py        - trainer to manage the model training
|   ├── 📂 model_trained       - model checkpoints and mlflow metrics
│   ├── 🐍 constants.py        - const values used through the project
|   ├── 🐍 model.py            - xgboost code
|   ├── 🐍 plots.py            - methods to create plot
|   └── 🐍 preprocess.py       - preprocess functions
├── 🐍 eda.ipynb               - the explotary data analysis
├── 🐍 nn.ipynb                - the nn in pytorch lightning
└── 🐍 xgboost.ipynb           - xboogst model results
```

## 📊 Results

There are three notebooks to explore to review the results.  The first one is `explotary_data_analysis.ipynb`, where you'll find the EDA, where we analyze the quality of the dataset and explore the features. It is a brief exploration given the limited time to complete it. A conclusion summarizes the key points at the end.

The second notebook is `model_xgboost.ipynb`, where we have the first model that uses a machine learning algorithm: XGBoost. It includes preprocessing, training, testing, and understanding the model's features and predictions using SHAP.

The third and final notebook is `model_neural_network.ipynb`, which explores a simple neural network, representing a deep learning approach. To make this method more original and demonstrate some software engineering skills, I used the PyTorch Lightning framework, and to track various metrics, I used MLflow. If you wish to run the model, it is recommended to do so on a GPU.

Additionally, to observe the metrics, it is necessary to start MLflow locally with the command in the terminal at the root of the folder: `mlflow ui --backend-store-uri ./model_trained/mlflow` and open the page http://127.0.0.1:5000.
