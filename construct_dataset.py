import numpy as np
import pandas as pd

FEATURE_PATH="extracted/extracted_audio.npy"
TRIMMED_FEATURE_PATH="extracted/trimmed_audio.npy"
LABEL_PATH="extracted/combined_position.csv"
OUTPUT_LABEL_PATH="extracted/trimmed_position.npy"

if __name__ == "__main__":
    features = np.load(FEATURE_PATH)
    labels = pd.read_csv(LABEL_PATH)

    selected_labels = []

    for feature in features:
        ts = feature[0] # removed the conversion from nanoseconds
        label_index = abs(labels["Time"].values - ts).argmin()
        selected_labels.append(labels.iloc[label_index:label_index+1].values.reshape(-1))
    
    droppable = np.where(features.sum(axis=0) != 0)

    label_array = np.stack(selected_labels)
    np.save(OUTPUT_LABEL_PATH, label_array)
    np.save(TRIMMED_FEATURE_PATH, features[:,:518])