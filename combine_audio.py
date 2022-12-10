import pandas as pd
import numpy as np
from os import listdir

SAMPLE_DIR="audio_samples_periodic_hf/"
OUTPUT_NAME="extracted/extracted_audio.npy"

if __name__ == "__main__":
    df = None
    for f in sorted(list(filter(lambda s: ".csv" in s, listdir(SAMPLE_DIR)))):
        new = np.genfromtxt(f"{SAMPLE_DIR}{f}",delimiter=",")
        print(f"{new.shape} new")
        if df is None:
            df = new
        else:
            df = np.concatenate([df, new])
            print(df.shape)
    np.save(OUTPUT_NAME, df)