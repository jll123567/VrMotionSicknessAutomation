from typing import Tuple, List

import tensorflow as tf
import tensorboard
import pandas as pd
import numpy as np
from keras import Model
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow import keras
import re
import decimal

DEBUG = False
period = int((5 * 60) / 3)  # (second*frames_per_seconds)/pooling_rate
downscale_ratio = 4  # How far to downscale images from original size, must be power of 2

# Round image dimensions similar to tf.decode_jpeg's rounding.
img_x = int(decimal.Decimal(525 / downscale_ratio).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
img_y = int(decimal.Decimal(1024 / downscale_ratio).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

img_size = (img_x, img_y)  # Size of images, they may need to be small(original 525,1024).


def dataset_dict_to_rows(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Remap tensor of tensor of (key, value) pairs to a tensor of values."""
    return dataset.map(lambda r: [v for _, v in r.items()], num_parallel_calls=tf.data.AUTOTUNE)


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

    # print("camera.csv[0:4]")
    # for row in camera.take(5):
    #     print(f"  {[item.numpy() for item in row]}")

    return camera


def load_control(recording_path: str) -> tf.data.Dataset:
    """Extract the controller data (button touch, button press, axis0-4x/y) from csv to a tf Dataset."""
    control_df = pd.read_csv(recording_path + "/control.csv", header=0, )
    control_df = control_df.drop(
        control_df[(control_df["framecounter"] % 3 != 0) | (control_df["framecounter"] == 0)].index)
    control_df = control_df.drop(["timestamp", "packetNum", "Unnamed: 16", "buttonPressed", "buttonTouched"], axis=1)
    control_tensor = tf.data.Dataset.from_tensor_slices(dict(control_df))

    def reformat_control(*control_row):
        """Unbatch and convert each device's values to one row."""
        new_row = []

        for i in range(2, 12):
            for j in range(3):
                new_row.append(control_row[i][j])

        return new_row

    control_tensor = dataset_dict_to_rows(control_tensor).batch(3).map(reformat_control,
                                                                       num_parallel_calls=tf.data.AUTOTUNE)

    # print("control.csv[0:4]")
    # for row in control_tensor.take(5):
    #     print(f"  {[item.numpy() for item in row]}")

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

    pose_tensor = dataset_dict_to_rows(pose_tensor).batch(3).map(reformat_pose, num_parallel_calls=tf.data.AUTOTUNE)

    # print("pose.csv[0:4]")
    # for row in pose_tensor.take(5):
    #     print(f"  {[item.numpy() for item in row]}")

    return pose_tensor


def load_numeric(recording_path: str) -> tf.data.Dataset:
    camera = load_camera(recording_path)
    control = load_control(recording_path)
    pose = load_pose(recording_path)

    numeric = tf.data.Dataset.zip(camera, control, pose)

    def unpack_numeric(cam: tf.Tensor, ctrl: tf.Tensor, pose: tf.Tensor) -> tf.Tensor:
        num = []

        for i in cam:
            num.append(i)
        for i in ctrl:
            num.append(i)
        for i in pose:
            num.append(i)

        return tf.convert_to_tensor(num, dtype=tf.float64)

    numeric = numeric.map(unpack_numeric, num_parallel_calls=tf.data.AUTOTUNE).batch(period)

    # print("Numeric[0:4]")
    # for row in numeric.take(5):
    #     print(f"  {[item.numpy() for item in row]}")

    return numeric


def load_voice(recording_path: str) -> tf.data.Dataset:
    """Extract voice(motion sickness rating) from csv to tf Dataset."""
    voice_df = pd.read_csv(recording_path + "/voice_preproc.csv", header=0)
    voice_df = voice_df.drop(["nearest_frame", "timestamp", "method"], axis=1)
    voice_tensor = tf.data.Dataset.from_tensor_slices(dict(voice_df))
    voice_tensor = dataset_dict_to_rows(voice_tensor)
    voice_tensor = voice_tensor.batch(period)

    @tf.function
    def reformat_voice(ratings):
        """Reduce the batch to just its mean, and represent mean rating as a one-hot tensor."""
        return tf.one_hot(tf.math.reduce_mean(ratings) - 1, 5)  # Ratings are [1,5], -1 to put in range [0,4]

    voice_tensor = voice_tensor.map(reformat_voice, num_parallel_calls=tf.data.AUTOTUNE)

    # print("voice_preproc.csv[0:4]")
    # for row in voice_tensor.take(5):
    #     print(f"  {[item.numpy() for item in row]}")

    return voice_tensor


def load_images(path: str) -> tf.data.Dataset:
    images_dataset = tf.data.Dataset.list_files(path + "/s*.jpg", shuffle=False)

    @tf.function
    def decode_img(img_fp: str) -> tf.Tensor:
        i = tf.io.read_file(img_fp)
        i = tf.io.decode_jpeg(i, ratio=downscale_ratio)
        i = tf.cast(i, tf.float32) / 255  # /255 so values are range [0,1]
        return i

    images_dataset = images_dataset.map(decode_img, num_parallel_calls=2).batch(period)

    # print("s#######[0]")
    # for img in images_dataset.take(1):
    #     i_np = img.numpy()
    #     print(f"  {i_np}")

    return images_dataset


def load_recording(path: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    y = load_voice(path)
    x_n = load_numeric(path)
    x_i = load_images(path)
    return tf.data.Dataset.zip(x_n, x_i), y


def load_dataset(paths: list[str]) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    x, y = load_recording(paths.pop(0))
    for i in paths:
        next_x, next_y = load_recording(i)
        x.concatenate(next_x)
        y.concatenate(next_y)
    return x, y


def test_train_split(x: tf.data.Dataset, y: tf.data.Dataset, split=0.8, batchsize=5) -> tuple:
    """
    Generate a train and test split for a dataset and convert rows to sets of the same number of steps.

    x and y must be the same length
    """
    l = int(x.cardinality().numpy())
    l_train = int(l * split)
    l_test = l - l_train

    x_train = x.take(l_train)
    x_test = x.take(l_test)
    y_train = y.take(l_train)
    y_test = y.take(l_test)

    return tf.data.Dataset.zip(x_train, y_train).shuffle(1000).batch(batchsize), tf.data.Dataset.zip(x_test, y_test).batch(batchsize)


def make_numeric_model(input_shape) -> tuple[Model, list[ModelCheckpoint | ReduceLROnPlateau | EarlyStopping]]:
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(5, activation="softmax")(
        gap)  # units is 5 since there are five classes(rating 1-5)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "checkpoints/best_model_numeric.keras", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)
    ]

    numeric_model = keras.Model(inputs=input_layer, outputs=output_layer, name="Numeric_Convolution_Model")
    print(numeric_model.summary())
    numeric_model.compile(optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["categorical_accuracy"]
                          )

    return numeric_model, callbacks


def make_image_model(input_shape) -> tuple[Model, list[ModelCheckpoint | ReduceLROnPlateau | EarlyStopping]]:
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling3D()(conv3)

    output_layer = keras.layers.Dense(5, activation="softmax")(
        gap)  # units is 5 since there are five classes(rating 1-5)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "checkpoints/best_model_image.keras", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)
    ]

    image_model = keras.Model(inputs=input_layer, outputs=output_layer, name="Image_Convolution_Model")
    print(image_model.summary())
    image_model.compile(optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=["categorical_accuracy"]
                        )

    return image_model, callbacks


def make_full_model(num_input_shape, img_input_shape) -> tuple[
    Model, list[ModelCheckpoint | ReduceLROnPlateau | EarlyStopping]]:
    num_input_layer = keras.layers.Input(num_input_shape)
    num_conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(num_input_layer)
    num_conv1 = keras.layers.BatchNormalization()(num_conv1)
    num_conv1 = keras.layers.ReLU()(num_conv1)
    num_conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(num_conv1)
    num_conv2 = keras.layers.BatchNormalization()(num_conv2)
    num_conv2 = keras.layers.ReLU()(num_conv2)
    num_conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(num_conv2)
    num_conv3 = keras.layers.BatchNormalization()(num_conv3)
    num_conv3 = keras.layers.ReLU()(num_conv3)
    num_gap = keras.layers.GlobalAveragePooling1D()(num_conv3)
    num_output_layer = keras.layers.Dense(5, activation="softmax")(num_gap)

    img_input_layer = keras.layers.Input(img_input_shape)
    img_conv1 = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same")(img_input_layer)
    img_conv1 = keras.layers.BatchNormalization()(img_conv1)
    img_conv1 = keras.layers.ReLU()(img_conv1)
    img_conv2 = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same")(img_conv1)
    img_conv2 = keras.layers.BatchNormalization()(img_conv2)
    img_conv2 = keras.layers.ReLU()(img_conv2)
    img_conv3 = keras.layers.Conv3D(filters=64, kernel_size=3, padding="same")(img_conv2)
    img_conv3 = keras.layers.BatchNormalization()(img_conv3)
    img_conv3 = keras.layers.ReLU()(img_conv3)
    img_gap = keras.layers.GlobalAveragePooling3D()(img_conv3)
    img_output_layer = keras.layers.Dense(5, activation="softmax")(img_gap)

    comb = keras.layers.Concatenate()([num_output_layer, img_output_layer])
    comb = keras.layers.Dense(32)(comb)
    comb = keras.layers.Dense(16)(comb)
    comb = keras.layers.Dense(8)(comb)
    full_out = keras.layers.Dense(5, activation="softmax")(comb)

    full_model = keras.Model(inputs=[num_input_layer, img_input_layer], outputs=full_out, name="Full_Model")
    print(full_model.summary())
    full_model.compile(optimizer="adam",
                       loss="categorical_crossentropy",
                       metrics=["categorical_accuracy"]
                       )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "checkpoints/3_6_24_model_full.keras", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        keras.callbacks.TensorBoard(log_dir='./logs/6_3_24_full', histogram_freq=1)
    ]

    return full_model, callbacks


if __name__ == "__main__":
    x, y = load_dataset([
        '/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/Pottery/P22 VRLOG-6061422',
        '/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/Epic_Roller_Coasters/P18 VRLOG-6051213',
        '/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/Mini_Motor_Racing/P23 VRLOG-6061400',
        '/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/VR_Rome/P9 VRLOG-5091805',
        '/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection/Beat_Saber/P4 VRLOG-5051047'
    ])

    train, test = test_train_split(x, y, split=0.8, batchsize=2)

    # numeric_model, numeric_callbacks = make_numeric_model((period, 116))
    # numeric_hist = numeric_model.fit(x=train,
    #                                  epochs=100,
    #                                  callbacks=numeric_callbacks,
    #                                  validation_data=test,
    #                                  verbose=1
    #                                  )
    # numeric_model.evaluate(test)

    # image_model, image_callbacks = make_image_model((period, img_size[0], img_size[1], 3))
    # image_hist = image_model.fit(x=train,
    #                                epochs=100,
    #                                callbacks=image_callbacks,
    #                                validation_data=test,
    #                                verbose=1
    #                                )
    # image_model.evaluate(test)

    full_model, full_callbacks = make_full_model((period, 116), (period, img_size[0], img_size[1], 3))
    full_hist = full_model.fit(x=train,
                               epochs=250,
                               callbacks=full_callbacks,
                               validation_data=test,
                               verbose=1
                               )
    full_model.evaluate(test)

    input("Training Finished. Press enter to quit.")
