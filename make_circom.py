import json

# Load weights and biases from the JSON file
with open('weights_biases.json', 'r') as f:
    data = json.load(f)

# Prepare Circom code
circom_code = """pragma circom 2.0.0;

template ReLU() {
    signal input x;
    signal output y;
    y <== (x < 0) * 0 + (x >= 0) * x;
}

template LinearLayer(num_inputs, num_outputs) {
    signal input inputs[num_inputs];
    signal input weights[num_inputs][num_outputs];
    signal input bias[num_outputs];
    signal output layer_output[num_outputs];  // Changed output to layer_output

    for (var j = 0; j < num_outputs; j++) {
        layer_output[j] <== bias[j];  // Updated assignment
        for (var i = 0; i < num_inputs; i++) {
            layer_output[j] <== layer_output[j] + inputs[i] * weights[i][j];  // Updated assignment
        }
    }
}

template NeuralNetwork() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;

    // Hidden Layer 1
    component layer1 = LinearLayer(3, 64);
    layer1.inputs[0] <== a;
    layer1.inputs[1] <== b;
    layer1.inputs[2] <== c;

    // Fill in weights and biases for Layer 1
"""
a = 10**17
# Fill in layer1 weights and biases
for i in range(64):
    for j in range(3):
        weight = data["layer1"]["weights"][j][i]
        circom_code += f"    layer1.weights[{j}][{i}] <== {int(weight*a)} / (10 ** 17);\n"  # Write division in the code
    bias = data["layer1"]["biases"][i]
    circom_code += f"    layer1.bias[{i}] <== {int(bias*a)} / (10 ** 17);\n"  # Write division in the code

circom_code += """
    // ReLU activation for Layer 1
    component relu1[64]; // One ReLU for each neuron in layer 1
    for (var j = 0; j < 64; j++) {
        relu1[j] = ReLU();
        relu1[j].x <== layer1.layer_output[j];  // Updated connection
    }

    // Hidden Layer 2
    component layer2 = LinearLayer(64, 32);
    for (var j = 0; j < 64; j++) {
        for (var k = 0; k < 32; k++) {
            layer2.inputs[j] <== relu1[j].y; // Connect ReLU outputs to Layer 2
        }
    }

    // Fill in weights and biases for Layer 2
"""

# Fill in layer2 weights and biases
for i in range(32):
    for j in range(64):
        weight = data["layer2"]["weights"][j][i]
        circom_code += f"    layer2.weights[{j}][{i}] <== {int(weight*a)} / (10 ** 17);\n"  # Write division in the code
    bias = data["layer2"]["biases"][i]
    circom_code += f"    layer2.bias[{i}] <== {int(bias*a)} / (10 ** 17);\n"  # Write division in the code

circom_code += """
    // ReLU activation for Layer 2
    component relu2[32]; // One ReLU for each neuron in layer 2
    for (var j = 0; j < 32; j++) {
        relu2[j] = ReLU();
        relu2[j].x <== layer2.layer_output[j];  // Updated connection
    }

    // Output Layer
    component outputLayer = LinearLayer(32, 1);
    for (var j = 0; j < 32; j++) {
        outputLayer.inputs[j] <== relu2[j].y; // Connect ReLU outputs to output layer
    }

    // Fill in weights and biases for Output Layer
"""

# Fill in output layer weights and biases
for i in range(1):  # Only one output neuron
    for j in range(32):
        weight = data["output_layer"]["weights"][j][i]
        circom_code += f"    outputLayer.weights[{j}][{i}] <== {int(weight*a)} / (10 ** 17);\n"  # Write division in the code
    bias = data["output_layer"]["biases"][i]
    circom_code += f"    outputLayer.bias[{i}] <== {int(bias*a)} / (10 ** 17);\n"  # Write division in the code

circom_code += """
    // Final output
    out <== outputLayer.layer_output[0];  // Updated output connection
}

component main = NeuralNetwork();
"""

# Write the Circom code to a file
with open('neural_network.circom', 'w') as f:
    f.write(circom_code)

print("Circom code has been generated and saved to 'neural_network.circom'.")