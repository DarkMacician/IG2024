import json

from main import nn_model


def save_weights_to_json(model, filename):
    weights = model.get_weights()
    weights_dict = {}

    for i, (W, b) in enumerate(zip(weights[::2], weights[1::2])):
        weights_dict[f'layer_{i + 1}'] = {
            'weights': W.tolist(),
            'biases': b.tolist()  # Convert numpy array to list
        }

    with open(filename, 'w') as f:
        json.dump(weights_dict, f, indent=4)


# Call the function to save weights
save_weights_to_json(nn_model, 'model_weights.json')

print("Weights and biases saved to model_weights.json.")