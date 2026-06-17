import json
import os
from datetime import datetime

LOG_FILE = 'predictions_log.json'

def log_prediction(input_data, prediction, probability):
    # Log entry banao
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "prediction": prediction,
        "probability": probability
    }

    # Existing logs load karo
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)

    # Naya entry add karo
    logs.append(log_entry)

    # Save karo
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)