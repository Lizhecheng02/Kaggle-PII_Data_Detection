## This Repo is for [Kaggle - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Download Dataset

```bash
cd combined_dataset
kaggle datasets download -d lizhecheng/pii-data-detection-dataset
unzip pii-data-detection-dataset.zip
```

```bash
cd kaggle_dataset
cd competition
kaggle competitions download -c pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip
```

