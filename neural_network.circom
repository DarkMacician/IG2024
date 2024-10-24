pragma circom 2.0.0;

include "./zk-models/circom_circuits/ReLU.circom";

// Matrix multiplication
template MatMul(M, N, P) {
    signal input A[M][N];
    signal input B[N][P];
    signal output C[M][P];

    signal intermediate[M][N][P];

    for (var i = 0; i < M; i++) {
        for (var j = 0; j < P; j++) {
            var sum = 0;
            for (var k = 0; k < N; k++) {
                intermediate[i][k][j] <== A[i][k] * B[k][j];
                sum += intermediate[i][k][j];
            }
            C[i][j] <== sum;
        }
    }
}

// Linear layer with bias
template LinearLayer(in_features, out_features) {
    signal input in[in_features];
    signal input weights[in_features][out_features];
    signal input bias[out_features];
    signal output out[out_features];

    component matmul = MatMul(1, in_features, out_features);
    for (var i = 0; i < in_features; i++) {
        matmul.A[0][i] <== in[i];
    }
    matmul.B <== weights;

    for (var i = 0; i < out_features; i++) {
        out[i] <== matmul.C[0][i] + bias[i];
    }
}

// Neural Network with two hidden layers
template NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size) {
    signal input x[input_size];
    signal input w1[input_size][hidden_size1];
    signal input b1[hidden_size1];
    signal input w2[hidden_size1][hidden_size2];
    signal input b2[hidden_size2];
    signal input w3[hidden_size2][output_size];
    signal input b3[output_size];
    signal output y[output_size];

    component layer1 = LinearLayer(input_size, hidden_size1);
    component relu1[hidden_size1];
    component layer2 = LinearLayer(hidden_size1, hidden_size2);
    component relu2[hidden_size2];
    component layer3 = LinearLayer(hidden_size2, output_size);

    // Scaling factor (1,000,000)
    var scale = 1000000;

    // First layer
    for (var i = 0; i < input_size; i++) {
        layer1.in[i] <== x[i] / scale;
    }
    layer1.weights <== w1;
    layer1.bias <== b1;

    // ReLU activation for first hidden layer
    for (var i = 0; i < hidden_size1; i++) {
        relu1[i] = ReLU();
        relu1[i].in <== layer1.out[i];
    }

    // Second layer
    for (var i = 0; i < hidden_size1; i++) {
        layer2.in[i] <== relu1[i].out;
    }
    layer2.weights <== w2;
    layer2.bias <== b2;

    // ReLU activation for second hidden layer
    for (var i = 0; i < hidden_size2; i++) {
        relu2[i] = ReLU();
        relu2[i].in <== layer2.out[i];
    }

    // Third layer (output layer)
    for (var i = 0; i < hidden_size2; i++) {
        layer3.in[i] <== relu2[i].out;
    }
    layer3.weights <== w3;
    layer3.bias <== b3;

    // Output (scale back the result)
    for (var i = 0; i < output_size; i++) {
        y[i] <== layer3.out[i] * scale;
    }
}

// Example usage: 3 input features, 64 neurons in first hidden layer, 32 neurons in second hidden layer, 1 output
component main {public [x]} = NeuralNetwork(3, 64, 32, 1);
