from main import nn_model
import pandas as pd
import json

df = pd.read()

public_data = {
    "model_architecture": {
        "layer1": {"units": 64, "activation": "relu"},
        "layer2": {"units": 32, "activation": "relu"},
        "layer3": {"units": 1, "activation": "linear"}
    },
    "hyperparameters": {
        "batch_size": 50,
        "epochs": 20,
        "learning_rate": 0.001
    },
    "loss_function": "mean_squared_error",
}

trained_weights = nn_model.get_weights()

private_witness = {
    "dataset": {
        "a": a,
        "b": b,
        "c": c,
        "y": y
    },
    "trained_weights": {
        "layer1": {
            "weights": trained_weights[0].tolist(),
            "biases": trained_weights[1].tolist()
        },
        "layer2": {
            "weights": trained_weights[2].tolist(),
            "biases": trained_weights[3].tolist()
        },
        "output_layer": {
            "weights": trained_weights[4].tolist(),
            "biases": trained_weights[5].tolist()
        },
    }
}

final_data = {
    "public_data": public_data,
    "private_witness": private_witness
}

with open('final_data.json', 'w') as f:
    json.dump(final_data, f, indent=4)

print("Public and private data saved to final_data.json.")
