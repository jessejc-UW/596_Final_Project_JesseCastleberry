import sys
import os
import requests

# Resolve project root relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from src.utils import evaluate_model, prepare_data
from src.model import HeartRateEstimator

MODEL_URL = f'https://raw.githubusercontent.com/jessejc-UW/596_Final_Project_JesseCastleberry/main/checkpoints/20251211-191633/heart_rate_estimator.pth'
TEST_DATA_URL = f'https://raw.githubusercontent.com/jessejc-UW/596_Final_Project_JesseCastleberry/main/data/test_data.npz'
TRAIN_DATA_URL = f'https://raw.githubusercontent.com/jessejc-UW/596_Final_Project_JesseCastleberry/main/data/train_data.npz'

MODEL_PATH = os.path.join(project_root, 'demo', 'tmp', 'heart_rate_estimator.pth')
TEST_DATA_PATH = os.path.join(project_root, 'demo', 'tmp', 'test_data.npz')
TRAIN_DATA_PATH = os.path.join(project_root, 'demo', 'tmp', 'train_data.npz')

DEMO_RESULTS_DIR = os.path.join(project_root, 'demo', 'results')

def download_file(url, save_path):
    response = requests.get(url)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Download model and data files
download_file(MODEL_URL, MODEL_PATH)
download_file(TEST_DATA_URL, TEST_DATA_PATH)
download_file(TRAIN_DATA_URL, TRAIN_DATA_PATH)

# Load trained model
model = HeartRateEstimator()
model.load_state_dict(torch.load(MODEL_PATH))

# Prepare data
train_loader, _, test_loader, _ = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH)

# Evaluate model
evaluate_model(model, test_loader, train_loader, results_dir=DEMO_RESULTS_DIR)