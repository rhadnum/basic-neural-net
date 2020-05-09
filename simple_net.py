
import pickle
import numpy as np
import random
#  Load picke data
with open("data/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

# Define data
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

# Define constants
image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig

def sigmoid_der(x):
        # We're not using this as our result has already been passed through signmoid at this point
        # return sigmoid(x) * (1 - sigmoid(x))
        return x * (1 - x)

class NeuralNetwork():
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        # Generate random weights for all input neurons
        # np.random.seed(1)

        # Set constants for neuron counts 
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = .1
        self.cost = 0

        # Generate matrix of weights for each output neuron ((* 2) -1 is used to generate between 1 and -1)
        self.h_bias = (np.random.rand(hidden_neurons, 1) * 2) - 1
        self.o_bias = (np.random.rand(output_neurons, 1) * 2) - 1
        self.ih_weights = (np.random.rand(hidden_neurons, input_neurons) * 2) - 1
        self.ho_weights = (np.random.rand(output_neurons, hidden_neurons) * 2) - 1

    def train(self, input, output):
        # Convert input to matrix for dot product
        input_matrix_t = np.asmatrix(input).T

        # Convert output to matrix for error
        output_matrix = np.asmatrix(output)

        # Generating activations for hidden layer
        hidden_weighted_sum = np.dot(self.ih_weights, input_matrix_t)
        hidden_weighted_sum_bias = np.add(hidden_weighted_sum, self.h_bias)
        hidden_activations = sigmoid(hidden_weighted_sum_bias)

         # Generating output result
        output_weighted_sum = np.dot(self.ho_weights, hidden_activations)
        output_weighted_sum_bias = output_weighted_sum + self.o_bias
        output_activations = sigmoid(output_weighted_sum_bias)

        # Print outputs  
        print(output_activations)
       
        # # Get errors for outputs
        output_errors = np.subtract(output_matrix, output_activations)
        self.cost += np.square(output_errors)
      
        # Transpose from [[w11,w12], [w21,w22]]
        # Transpose into [[w11,w21],[w12,w22]] 
        weights_ho_t = np.asarray(self.ho_weights).T
        hidden_errors = np.dot(weights_ho_t, output_errors)

        # Calculate gradients for ho layer
        output_activations_der = np.vectorize(sigmoid_der)(output_activations)
        gradients = np.multiply(output_activations_der, output_errors) * self.learning_rate

        # Calculate deltas
        hidden_t = np.asarray(hidden_activations).T
        weight_ho_deltas = np.multiply(gradients, hidden_t)

        # # Elementwise add weights with deltas 
        self.ho_weights = np.add(self.ho_weights, weight_ho_deltas)

        # # Adjust bias by gradients
        self.o_bias = np.add(self.o_bias, gradients)

        # Calculate gradients for hidden layer - this is the steepness of the drop for the deltas
        hidden_activations_der = np.vectorize(sigmoid_der)(hidden_activations)
        hidden_gradient = np.multiply(hidden_activations_der, hidden_errors) * self.learning_rate

        # Calculate ih deltas - These are the differences to make to the weights
        weight_ih_deltas = np.multiply(hidden_gradient, input)

        # Elementwise add weights with deltas 
        self.ih_weights = np.add(self.ih_weights, weight_ih_deltas)

        # Adjust bias by gradients 
        self.h_bias = np.add(self.h_bias, hidden_gradient) 

    def predict(self, input):
        # Convert input to matrix for dot product
        input_matrix_t = np.asmatrix(input).T

        # Generating activations for hidden layer
        hidden_weighted_sum = np.dot(self.ih_weights, input_matrix_t)
        hidden_weighted_sum_bias = np.add(hidden_weighted_sum, self.h_bias)
        hidden_activations = sigmoid(hidden_weighted_sum_bias)

         # Generating output result
        output_weighted_sum = np.dot(self.ho_weights, hidden_activations)
        output_weighted_sum_bias = np.add(output_weighted_sum, self.o_bias)
        hidden_activations = sigmoid(output_weighted_sum_bias)
        
        print(np.array(hidden_activations).flatten())

nn = NeuralNetwork(2,16,1)

inputs = [[0,1], [1,0], [0,0], [1,1]]
outputs = [[0], [0], [1], [1]]

# Train on a XOR
for idx in range(15000):
    data = random.choice(inputs)
    index = inputs.index(data)
    nn.train(data, outputs[index])

# Print predicted values
print('')
print('Predicted:')
nn.predict([0,0])
nn.predict([1,0])
nn.predict([0,1])
nn.predict([1,1])

print()
print('Cost:')
print(1 / 50000 * nn.cost )