import json
from python_snarks import 
import numpy as np

# Load data from JSON file
with open('validate_data.json', 'r') as f:
    data = json.load(f)

# Extract public data and private witness
public_data = data['public_data']
private_witness = data['private_witness']

# Public values: batch size, learning rate, etc.
batch_size = PubVal(public_data['hyperparameters']['batch_size'])
learning_rate = PubVal(public_data['hyperparameters']['learning_rate'])
epochs = PubVal(public_data['hyperparameters']['epochs'])
predicted_outputs = np.array(public_data['predicted_outputs'])
loss_function = public_data['loss_function']  # Used later to validate loss

# Private values: dataset and trained weights
a = np.array(private_witness['dataset']['a'])
b = np.array(private_witness['dataset']['b'])
c = np.array(private_witness['dataset']['c'])
y = np.array(private_witness['dataset']['y'])

weights = np.array(private_witness['trained_weights']['weights'])
biases = np.array(private_witness['trained_weights']['biases'])

# Convert to private values (PrivVal) for zk-SNARK proof
priv_a = [PrivVal(x) for x in a]
priv_b = [PrivVal(x) for x in b]
priv_c = [PrivVal(x) for x in c]
priv_y = [PrivVal(x) for x in y]
priv_weights = [PrivVal(w) for w in weights.flatten()]
priv_biases = [PrivVal(bias) for bias in biases]

# Define a function to model the neural network layers
def forward_pass(a, b, c, weights, biases):
    X = np.stack([a, b, c], axis=1)
    out_layer1 = np.maximum(0, np.dot(X, weights) + biases)  # ReLU activation for first layer
    out_layer2 = np.dot(out_layer1, weights) + biases  # No activation for the final layer
    return out_layer2

# Perform forward pass using private witness (weights, biases)
output = forward_pass(priv_a, priv_b, priv_c, priv_weights, priv_biases)

# Validate prediction with zk-SNARK
for i in range(len(output)):
    assert snark.abs(output[i] - PrivVal(predicted_outputs[i])) < 1e-5, "Prediction mismatch"

# Verify loss function (mean squared error)
mse_loss = snark.sum((PrivVal(y[i]) - output[i]) ** 2 for i in range(len(y))) / len(y)
print(f"Loss: {mse_loss.reveal()}")

# Run the PySNARK runtime to generate proof
print("Running zk-SNARK proof generation...")
