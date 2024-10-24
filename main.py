import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json

def generate_data(num_samples):
    a = np.random.rand(num_samples).tolist()
    b = np.random.rand(num_samples).tolist()
    c = np.random.rand(num_samples).tolist()
    y = (2 * np.array(a) + 3 * np.array(b) + 4 * np.array(c)).tolist()
    return a, b, c, y

num_samples = 10000
a, b, c, y = generate_data(num_samples)
data = {
    "a": a,
    "b": b,
    "c": c,
    "y": y
}

with open('data.json', 'w') as f:
    json.dump(data, f)

class NeuralNetworkModel:
    def __init__(self, input_dim=3, learning_rate=0.0001):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu', name='layer1'))
        self.model.add(Dense(32, activation='relu', name='layer2'))
        self.model.add(Dense(1, activation='linear', name='output_layer'))
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    def train(self, X, Y, epochs=20, batch_size=50):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def load_weights_from_json(self, filename):
        with open(filename, 'r') as f:
            weights_dict = json.load(f)
        weights_to_set = []
        for layer in weights_dict.values():
            weights_to_set.append(np.array(layer['weights']))
            weights_to_set.append(np.array(layer['biases']))
        self.set_weights(weights_to_set)

    def predict(self, X):
        return self.model.predict(X)

X = np.column_stack((a, b, c))

nn_model = NeuralNetworkModel(input_dim=3)
nn_model.train(X, y, epochs=20, batch_size=50)

trained_weights = nn_model.get_weights()

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
    "predicted_outputs": nn_model.predict(X).tolist(),
    "loss_function": "mean_squared_error",
}

private_witness = {
    "dataset": {
        "a": a.to_list(),
        "b": b.to_list(),
        "c": c.to_list(),
        "y": y.to_list()
    },
    "trained_weights": {
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
        },
    }
}

final_data = {
    "public_data": public_data,
    "private_witness": private_witness
}

with open('validate_data.json', 'w') as f:
    json.dump(final_data, f, indent=4)

print("Public and private data saved to final_data.json.")
