from bagpy import bagreader
from os.path import exists
from os import listdir
import pandas as pd

BAG_DIR="bagfiles/"
OUTPUT_DIR="extracted/"
POSE_TOPIC="/mocap_node/drone/pose"

if __name__ == "__main__":

    position_csvs = []

    for f in filter(lambda s: ".bag" in s, listdir(BAG_DIR)):

        output_name = f"{OUTPUT_DIR}{f.replace('.bag', '.csv')}"

        if not exists(output_name):

            bg = bagreader(f"{BAG_DIR}{f}")

            position_csvs.append(bg.message_by_topic(POSE_TOPIC))

    df = pd.read_csv(position_csvs[0])

    for pcsv in position_csvs[1:]:
        pd.concat([df, pd.read_csv(pcsv)])

    cols = [
        "Time",
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
    ]

    df.sort_values("Time")
    df[cols].to_csv(f"{OUTPUT_DIR}combined_position.csv",index=False)
