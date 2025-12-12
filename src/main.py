from model import *
from utils import *
import time

run_timestamp = time.strftime("%Y%m%d-%H%M%S")
DATA_DIR = Path('./data')
MODEL_DIR = Path('./checkpoints') / run_timestamp
RESULTS_DIR = Path('./results') / run_timestamp
TRAIN_FILE = DATA_DIR / 'train_data.npz'
TEST_FILE = DATA_DIR / 'test_data.npz'

def main():
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_data(TRAIN_FILE, TEST_FILE)

    # Initialize model
    model, device = init_model()

    # Train model
    trained_model, metrics = train_model(model=model, 
                                         train_loader=train_loader, 
                                         val_loader=val_loader, 
                                         epochs=1000,
                                         lr=5e-5,
                                         device=device, 
                                         save_model=True, 
                                         checkpoint_dir=MODEL_DIR, 
                                         results_dir=RESULTS_DIR,
                                         verbose=True)

    # Evaluate model
    evaluate_model(trained_model, test_loader, train_loader, device=device, results_dir=RESULTS_DIR)

if __name__ == "__main__":
    main()
