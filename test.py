import json

import numpy as np

from data_making import load_data_from_json
from model import NeuralNetworkModel

a, b, c, y = load_data_from_json()
a, b, c, y = np.array(a), np.array(b), np.array(c), np.array(y)

X = np.column_stack((a, b, c))

nn_model = NeuralNetworkModel(input_dim=3)
nn_model.train(X, y, epochs=20, batch_size=50)
trained_weights = nn_model.get_weights()

data = {
    "layer1": {
        "weights": trained_weights[0][0].tolist(),
        "biases": trained_weights[1][0].tolist()
    },
    "layer2": {
        "weights": trained_weights[0][1].tolist(),
        "biases": trained_weights[1][1].tolist()
    },
    "output_layer": {
        "weights": trained_weights[0][2].tolist(),
        "biases": trained_weights[1][2].tolist()
    }
}

# Save the dictionary to a JSON file
with open('weights_biases.json', 'w') as f:
    json.dump(data, f, indent=4)  # Pretty-print with an indent of 4 spaces

print("Weights and biases saved to 'weights_biases.json'.")