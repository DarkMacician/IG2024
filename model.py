# import json
# import numpy as np
# from keras import Sequential
# from keras.src.layers import Dense
# from keras.src.optimizers import Adam
#
#
# class NeuralNetworkModel:
#     def __init__(self, input_dim=3):
#         self.model = Sequential()
#         self.model.add(Dense(64, input_dim=input_dim, activation='relu', name='layer1'))
#         self.model.add(Dense(32, activation='relu', name='layer2'))
#         self.model.add(Dense(1, name='output_layer'))
#         self.model.compile(optimizer=Adam(), loss='mean_squared_error')
#
#     def train(self, X, Y, epochs=20, batch_size=50):
#         self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)
#
#     def get_weights_as_json(self):
#         # Get the weights and biases of each layer
#         weights = self.model.get_weights()
#
#         weights_dict = {
#             "w1": weights[0][0].tolist(),  # Weights for the first layer
#             "b1": weights[1][0].tolist(),  # Biases for the first layer
#             "w2": weights[0][1].tolist(),  # Weights for the second layer
#             "b2": weights[1][1].tolist(),  # Biases for the second layer
#             "w3": weights[0][2].tolist(),  # Weights for the output layer
#             "b3": weights[1][2].tolist()  # Biases for the output layer
#         }
#
#         return weights_dict
#
#     def save_weights_to_json(self, filename):
#         """Save weights and biases to a JSON file."""
#         weights_dict = self.get_weights_as_json()
#
#         # Write to file
#         with open(filename, 'w') as f:
#             json.dump(weights_dict, f, indent=4)
#
#     def load_weights_from_json(self, filename):
#         """Load weights and biases from a JSON file."""
#         with open(filename, 'r') as f:
#             weights_dict = json.load(f)
#
#         # Convert weights and biases back to numpy arrays and set them to the model
#         weights_to_set = [
#             np.array(weights_dict['w1']),
#             np.array(weights_dict['b1']),
#             np.array(weights_dict['w2']),
#             np.array(weights_dict['b2']),
#             np.array(weights_dict['w3']),
#             np.array(weights_dict['b3'])
#         ]
#
#         self.model.set_weights(weights_to_set)
#
#     def predict(self, X):
#         Y_pred = self.model.predict(X)
#         return Y_pred
#
#     def compute_loss(self, Y, Y_pred):
#         loss = np.mean((Y - Y_pred) ** 2)
#         return loss
#
#     def save_weights_to_json_custom(self, filename, input_vector=None):
#         # Get weights and biases in the specified format
#         weights_dict = self.get_weights_as_json()
#
#         # If you want to save the input vector 'x', pass it as input_vector
#         if input_vector is not None:
#             weights_dict['x'] = input_vector.tolist() if isinstance(input_vector, np.ndarray) else input_vector
#
#         # Save to a JSON file
#         with open(filename, 'w') as f:
#             json.dump(weights_dict, f, indent=4)
#
#
# def load_variables_from_json(filename):
#     """Load variables a, b, c, and y from a JSON file."""
#     with open(filename, 'r') as f:
#         data = json.load(f)
#
#     # Assuming the file contains keys 'a', 'b', 'c', and 'y'
#     a = np.array(data['a'])  # Input vector
#     b = np.array(data['b'])  # Weights for the first layer (example)
#     c = np.array(data['c'])  # Another variable (could be biases or any other value)
#     y = np.array(data['y'])  # Target output vector
#
#     return a, b, c, y
#
# nn = NeuralNetworkModel()
# a, b, c, y = load_variables_from_json("data.json")
# X = np.stack([a[:10], b[:10], c[:10]], axis=0)
# y = y[:10]
#
# nn.save_weights_to_json_custom("witness.json", X)
#
#
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam


class NeuralNetworkModel:
    def __init__(self, input_dim=3):
        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))  # Input layer
        self.model.add(Dense(64, activation='relu', name='layer1'))
        self.model.add(Dense(32, activation='relu', name='layer2'))
        self.model.add(Dense(1, name='output_layer'))
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

    def train(self, X, Y, epochs=20, batch_size=50):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

    def get_weights_as_json(self):
        # Get the weights and biases of each layer
        weights = self.model.get_weights()
        scale_factor = 10**17

        weights_dict = {
            "w1": (weights[0] * scale_factor).astype(np.int64).tolist(),  # Scale weights for the first layer
            "b1": (weights[1] * scale_factor).astype(np.int64).tolist(),  # Scale biases for the first layer
            "w2": (weights[2] * scale_factor).astype(np.int64).tolist(),  # Scale weights for the second layer
            "b2": (weights[3] * scale_factor).astype(np.int64).tolist(),  # Scale biases for the second layer
            "w3": (weights[4] * scale_factor).astype(np.int64).tolist(),  # Scale weights for the output layer
            "b3": (weights[5] * scale_factor).astype(np.int64).tolist()   # Scale biases for the output layer
        }

        return weights_dict

    def save_weights_to_json(self, filename):
        """Save weights and biases to a JSON file."""
        weights_dict = self.get_weights_as_json()
        # Write to file
        with open(filename, 'w') as f:
            json.dump(weights_dict, f, indent=4)

    def load_weights_from_json(self, filename):
        """Load weights and biases from a JSON file."""
        with open(filename, 'r') as f:
            weights_dict = json.load(f)

        scale_factor = 10**17
        # Convert weights and biases back to numpy arrays and set them to the model
        weights_to_set = [
            np.array(weights_dict['w1'], dtype=np.float32) / scale_factor,
            np.array(weights_dict['b1'], dtype=np.float32) / scale_factor,
            np.array(weights_dict['w2'], dtype=np.float32) / scale_factor,
            np.array(weights_dict['b2'], dtype=np.float32) / scale_factor,
            np.array(weights_dict['w3'], dtype=np.float32) / scale_factor,
            np.array(weights_dict['b3'], dtype=np.float32) / scale_factor
        ]
        self.model.set_weights(weights_to_set)

    def predict(self, X):
        Y_pred = self.model.predict(X)
        return Y_pred

    def compute_loss(self, Y, Y_pred):
        loss = np.mean((Y - Y_pred) ** 2)
        return loss

    def save_weights_to_json_custom(self, filename, input_vector=None):
        # Get weights and biases in the specified format
        weights_dict = self.get_weights_as_json()

        # If you want to save the input vector 'x', pass it as input_vector
        if input_vector is not None:
            weights_dict['x'] = input_vector.tolist() if isinstance(input_vector, np.ndarray) else input_vector

        # Save to a JSON file
        with open(filename, 'w') as f:
            json.dump(weights_dict, f, indent=4)


def load_variables_from_json(filename):
    """Load variables a, b, c, and y from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    # Ensure that the JSON contains keys for 'a', 'b', 'c', and 'y'
    a = np.array(data['a'])  # Input vector
    b = np.array(data['b'])  # Weights or other variable
    c = np.array(data['c'])  # Another variable (could be biases or any other value)
    y = np.array(data['y'])  # Target output vector

    return a, b, c, y


# Example usage
nn = NeuralNetworkModel()

a, b, c, y = load_variables_from_json("data.json")
X = np.stack([a, b, c], axis=0).T

nn.train(X, y)