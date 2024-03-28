import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
import keras.src.models.functional

import socket

import struct

model_path = "./saved_models/3_5_24_model_full.keras"
numeric_vals_amount = 100 * 116
image_vals_amount = 100 * 131 * 256 * 3


if __name__ == "__main__":
    # Load Model
    model = models.load_model(model_path)
    # Wait for connections
    host = "127.0.0.1"
    port = 9696

    with socket.socket() as sock:
        sock.bind((host, port))
        sock.listen()
        # Accept Connection
        while True:  # Continuously check for new connections.
            conn, addr = sock.accept()
            with conn:
                print(f"Connection from {addr}")
                conn_bytes = b""
                try:
                    while True:  # YOU NEED TO BREAK THIS CONNECTION MANUALLY!!!
                        # Receive numeric, image
                        data = conn.recv((4 * numeric_vals_amount) + (4 * image_vals_amount))  # recv all the bytes needed.
                        conn_bytes += data

                        # Check all required data received.
                        if not len(conn_bytes) == ((4 * numeric_vals_amount) + (4 * image_vals_amount)):
                            continue
                        else:
                            # Reformat data
                            numeric_bytes = conn_bytes[0:4 * numeric_vals_amount]
                            image_bytes = conn_bytes[4 * numeric_vals_amount:]
                            numeric_vals = np.array(struct.unpack('f' * numeric_vals_amount, numeric_bytes),  # Unpack from bytes to floats, put into a numpy array, then make sure it's the right shape.
                                                    np.float32).reshape((1, 100, 116))
                            image_vals = np.array(struct.unpack('f' * image_vals_amount, image_bytes), np.float32).reshape(
                                (1, 100, 131, 256, 3))
                            # inference
                            prediction = model.predict([numeric_vals, image_vals])
                            # Get most likely class.
                            largest_class = 0
                            for i in range(5):
                                if prediction[0][i] > prediction[0][largest_class]:  # Prediction is ndarray(1,5)
                                    largest_class = i
                            print(f"Prediction: {prediction}, {largest_class}")
                            # send result(as int)
                            conn.sendall(struct.pack("i", largest_class))  # May need to convert endian-ness in C#.
                            break  # Break loop and end connection.
                # error handling
                except TypeError as e:
                    print(f"Badly caught exception: {e}\nHandle this properly pls :)")
