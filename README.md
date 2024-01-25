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
sudo apt install unzip
```

```bash
kaggle datasets download -d lizhecheng/pii-data-detection-dataset
unzip pii-data-detection-dataset.zip
```
