import tensorflow as tf
import tensorboard
import pandas as pd
import numpy as np
from tensorflow import keras
import re

DEBUG = True


def dataset_dict_to_rows(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Remap tensor of tensor of (key, value) pairs to a tensor of values."""
    return dataset.map(lambda r: [v for _, v in r.items()])


def load_camera(recording_path: str) -> tf.data.Dataset:
    """Extract the camera data(cameraname, projection martix, view matrix) from csv to a tf Dataset."""
    camera_df = pd.read_csv(recording_path + "/camera.csv", header=0)

    # Split projection/view matrix string into sixteen columns.
    camera_df[
        ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16']] = \
        camera_df[
            'projection'].str.split(' ', n=15, expand=True)
    camera_df[
        ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16']] = \
        camera_df[
            'view'].str.split(' ', n=15, expand=True)

    # Convert project/view matrix to floats.
    camera_df = camera_df.astype(
        {'p1': float, 'p2': float, 'p3': float, 'p4': float, 'p5': float, 'p6': float, 'p7': float, 'p8': float,
         'p9': float, 'p10': float, 'p11': float, 'p12': float, 'p13': float, 'p14': float, 'p15': float,
         'p16': float, 'v1': float, 'v2': float, 'v3': float, 'v4': float, 'v5': float, 'v6': float, 'v7': float,
         'v8': float, 'v9': float, 'v10': float, 'v11': float, 'v12': float, 'v13': float, 'v14': float,
         'v15': float, 'v16': float})

    # Drop uneccesary columns/rows.
    camera_df = camera_df.drop(["projection", "view", "framecounter", "timestamp"], axis=1)
    if re.search(r'Pottery', recording_path):
        camera_df = camera_df.drop(camera_df[~camera_df["name"].str.contains(r'^Eye')].index)
    else:
        camera_df = camera_df.drop(camera_df[~camera_df["name"].str.contains(r'[Mm]ain|\(eye\)$|^Camera$')].index)
    camera_df = camera_df.drop(["name"], axis=1)

    # Convert to tensor in correct format.
    camera = tf.data.Dataset.from_tensor_slices(dict(camera_df))
    camera = dataset_dict_to_rows(camera)

    if DEBUG:
        print("camera.csv[0:4]")
        for row in camera.take(5):
            print(f"  {[item.numpy() for item in row]}")
    return camera


def load_control(recording_path: str) -> tf.data.Dataset:
    """Extract the controller data (button touch, button press, axis0-4x/y) from csv to a tf Dataset."""
    control_df = pd.read_csv(recording_path + "/control.csv", header=0, )
    control_df = control_df.drop(
        control_df[(control_df["framecounter"] % 3 != 0) | (control_df["framecounter"] == 0)].index)
    control_df = control_df.drop(["timestamp", "packetNum", "Unnamed: 16"], axis=1)
    control_tensor = tf.data.Dataset.from_tensor_slices(dict(control_df))

    def reformat_control(*control_row):
        """Unbatch and convert each device's values to one row."""
        new_row = []

        for i in range(2, 14):
            for j in range(3):
                new_row.append(control_row[i][j])

        return new_row

    control_tensor = dataset_dict_to_rows(control_tensor).batch(3).map(reformat_control)

    if DEBUG:
        print("control.csv[0:4]")
        for row in control_tensor.take(5):
            print(f"  {[item.numpy() for item in row]}")

    return control_tensor


def load_pose(recording_path: str) -> tf.data.Dataset:
    """Extract the controller data (button touch, button press, axis0-4x/y) from csv to a tf Dataset."""
    pose_df = pd.read_csv(recording_path + "/pose.csv", header=0, )
    pose_df = pose_df.drop(
        pose_df[(pose_df["framecounter"] % 3 != 0) | (pose_df["framecounter"] == 0)].index)
    pose_df = pose_df.drop(["timestamp", ], axis=1)

    # Expand transformMatrix, and velocities.
    pose_df[['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']] = \
        pose_df['deviceToAbsoluteTracking'].str.split(' ', n=12, expand=True)
    pose_df[['v1', 'v2', 'v3']] = \
        pose_df['velocity'].str.split(' ', n=3, expand=True)
    pose_df[['av1', 'av2', 'av3']] = \
        pose_df['angularVelocity'].str.split(' ', n=3, expand=True)
    pose_df = pose_df.drop(["deviceToAbsoluteTracking", "velocity", "angularVelocity"], axis=1)

    pose_df = pose_df.astype(
        {'m1': float, 'm2': float, 'm3': float,
         'm4': float, 'm5': float, 'm6': float,
         'm7': float, 'm8': float, 'm9': float,
         'm10': float, 'm11': float, 'm12': float,

         'v1': float, 'v2': float, 'v3': float,
         'av1': float, 'av2': float, 'av3': float})

    pose_tensor = tf.data.Dataset.from_tensor_slices(dict(pose_df))

    def reformat_pose(*pose_row):
        """Unbatch and convert each device's values to one row."""
        new_row = []

        for i in range(2, 20):
            for j in range(3):
                new_row.append(pose_row[i][j])

        return new_row

    pose_tensor = dataset_dict_to_rows(pose_tensor).batch(3).map(reformat_pose)

    if DEBUG:
        print("pose.csv[0:4]")
        for row in pose_tensor.take(5):
            print(f"  {[item.numpy() for item in row]}")

    return pose_tensor


def load_numeric(recording_path: str):
    camera = load_camera(recording_path)
    control = load_control(recording_path)
    pose = load_pose(recording_path)

    numeric = tf.data.Dataset.zip(camera, control, pose)

    if DEBUG:
        print("Numeric[0:4]")
        for row in numeric.take(5):
            print(f"  {[[element.numpy() for element in item] for item in row]}")

    return numeric


def load_voice(recording_path: str) -> tf.data.Dataset:
    """Extract voice(motion sickness rating) from csv to tf Dataset."""
    voice_col_types = [tf.int32]
    voice = tf.data.experimental.CsvDataset(recording_path + "/voice_preproc.csv", voice_col_types,
                                            select_cols=[2], header=True)
    if DEBUG:
        print("voice_preproc.csv[0:4]")
        for row in voice.take(5):
            print(f"  {[item.numpy() for item in row]}")
    return voice


def load_images(path: str):
    images_dataset = tf.data.Dataset.list_files(path+"/s*.jpg", shuffle=False)

    def decode_img(img_fp: str) -> tf.Tensor:
        i = tf.io.read_file(img_fp)
        i = tf.io.decode_image(i, dtype=tf.float32)
        return i

    images_dataset = images_dataset.map(decode_img)

    if DEBUG:
        print("s#######[0]")
        for img in images_dataset.take(1):
            print(f"  {img.numpy()}")

    return images_dataset


def load_recording(path: str):
    y = load_voice(path)
    X_n = load_numeric(path)
    X_i = load_images(path)
    return tf.data.Dataset.zip(X_n, X_i, y)


def load_dataset():
    pass


if __name__ == "__main__":
    images = load_images('/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/Pottery/P22 VRLOG-6061422')
    pass
