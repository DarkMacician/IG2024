# import numpy as np
# import json
#
# def generate_data(num_samples):
#     a = np.random.rand(num_samples).tolist()
#     b = np.random.rand(num_samples).tolist()
#     c = np.random.rand(num_samples).tolist()
#     y = (2 * np.array(a) + 3 * np.array(b) + 4 * np.array(c)).tolist()
#     return a, b, c, y
#
# def save_data_to_json(a, b, c, y):
#     data = {
#         "a": a,
#         "b": b,
#         "c": c,
#         "y": y
#     }
#     with open('data.json', 'w') as f:
#         json.dump(data, f)
#
# # Generate and save data
# num_samples = 10000
# a, b, c, y = generate_data(num_samples)
# save_data_to_json(a, b, c, y)
# print("Data generated and saved to data.json")
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json


# Create and train a neural network model
class NeuralNetworkModel:
    def __init__(self, input_dim=3, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    def train(self, X, Y, epochs=20, batch_size=50):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

    def get_weights(self):
        return self.model.get_weights()

    def predict(self, X):
        return self.model.predict(X)

def load_data_from_json():
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data["a"], data["b"], data["c"], data["y"]

def main():
    a, b, c, y = load_data_from_json()
    X = np.column_stack((a, b, c))
    print(type(X))

    nn_model = NeuralNetworkModel(input_dim=3)
    nn_model.train(X, y)

    # Get weights after training
    trained_weights = nn_model.get_weights()
    print("Trained weights and biases:", trained_weights)

    # Make predictions
    Y_pred = nn_model.predict(X)
    print("First 5 predicted outputs:", Y_pred[:5])


if __name__ == "__main__":
    main()
