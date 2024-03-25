import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models

import socket

model_path = "./3_5_24_model_full"

# This is idea implemented badly, but my sins have been pardoned by C# saints(literallyjustsmith, hordini).
if __name__ == "__main__":
    # Load Model
    model = models.load_model(model_path)
    # Wait for connections
    host = "127.0.0.1"
    port = 96969

    with socket.socket() as sock:
        sock.bind((host, port))
        sock.listen()
        # Accept Connection
        conn, addr = sock.accept()
        with conn:
            print(f"Connection from {addr}")
            conn_bytes = b""
            try:
                while True:  # YOU NEED TO BREAK THIS CONNECTION MANUALLY!!!
                    # Receive numeric, image
                    data = conn.recv(1024)  # increase this so you don't need to read multiple times.
                    conn_bytes += data

                    # Check all required data received.
                    is_required_data_present = False
                    if not is_required_data_present:
                        continue
                    else:
                        # Reformat data
                        numeric_bytes = conn_bytes[0:116]
                        image_bytes = conn_bytes[116:]
                        numeric_vals = []
                        image_vals = []
                        full_tensor = None
                        # inference
                        prediction = model.predict(full_tensor)
                        # Get most likely class.
                        largest_class = 0
                        for i in range(5):
                            if prediction[i] > prediction[largest_class]:
                                largest_class = i
                        # send result(as int)
                        conn.sendall(largest_class.to_bytes(4, 'big'))  # May need to convert endian-ness in C#.
                        break  # Break loop and end connection.
            # error handling
            except Exception as e:
                print(f"Badly caught exception: {e}\nHandle this properly pls :)")
