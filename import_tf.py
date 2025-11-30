import os
import sys
import time

import tensorflow as tf
from keras import Model, layers, mixed_precision, utils

if __name__ == '__main__':
    print("\nStarted\n")
    
    print("PYTHON", sys.executable)
    print("TF version:", tf.__version__)

    @tf.function
    def f(x):
        return x * 2

    cf = f.get_concrete_function(tf.constant(1)) # type: ignore
    print("function:", type(cf))

    inputs = layers.Input(shape=(1,))
    outputs = layers.Dense(1)(inputs)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    pred = model.predict(tf.constant([[2.0]]), verbose='0')
    print(f"prediction: {pred}")
    try:
        policy = mixed_precision.global_policy()
        print(f"mixed precision: {policy.name}")
    except Exception as e:  # noqa: BLE001
        print(f"! Mixed precision check failed: {e!s}")
    
    print("Done")

    print("\n Now sleeping...")
    time.sleep(60)
