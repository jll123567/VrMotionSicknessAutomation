import tensorflow as tf
import tensorboard
import pandas as pd
import numpy as np
from tensorflow import keras

DEBUG = True


def load_camera(recording_path: str):
    camera_df = pd.read_csv(recording_path+"/camera.csv", header=0)
    camera_df[
        ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16']] = camera_df[
        'projection'].str.split(' ', n=15, expand=True)
    camera_df[
        ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16']] = camera_df[
        'view'].str.split(' ', n=15, expand=True)
camera_df.drop(["projection", "view", "framecounter", "timestamp"], axis=1, inplace=True)
    return camera


def load_numeric(recording_path: str):
    num_files = ["camera.csv", "control.csv", "light.csv", "object.csv", "pose.csv"]


def load_voice(recording_path: str):
    voice_col_types = [tf.int32]
    voice = tf.data.experimental.CsvDataset(recording_path + "/voice_preproc.csv", voice_col_types,
                                            select_cols=[2], header=True)
    if DEBUG:
        print("voice_preproc.csv[0:4]")
        for line in voice.take(5):
            print(f"  {[item.numpy() for item in line]}")
    return voice


def load_image(path: str):
    pass


def load_recording(path: str):
    pass


def load_dataset():
    pass


if __name__ == "__main__":
    voice = load_camera(
        '/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/Epic_Roller_Coasters/P20 VRLOG-6051805')
    pass
