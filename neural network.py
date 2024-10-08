#!/usr/bin/env python
# coding: utf-8

# # Dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from typing import Tuple
from tqdm.notebook import tqdm
from itertools import product
from sklearn.preprocessing import StandardScaler


# # Neural Network Class

# In[2]:


class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int]) -> None:
        # Initialize the neural network with the given parameters.

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.hidden_layers: List[int] = hidden_layers

        # Create a complete list of layer sizes including input and output layers
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights and biases
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))  
      
        

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, loss_threshold: float = None) -> None:
        # Train the neural network using stochastic gradient descent with early stopping and data shuffling.
        
        num_samples = X.shape[0]
        total_iterations = epochs * num_samples      
        with tqdm(total=total_iterations, desc="Training Progress") as pbar:
            for epoch in range(epochs):
                # Shuffle data at the beginning of each epoch
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]
    
                epoch_loss = 0
                for i in range(num_samples):
                    x = X[i]
                    target = y[i]
                    
                    # Forward pass
                    activations = self.forward(x)
                    y_pred = activations[-1]
                    
                    # Compute loss
                    loss = self.compute_loss(target, y_pred)
                    epoch_loss += loss
                    
                    # Backward pass
                    grads_w, grads_b = self.backward(activations, target)
                    
                    # Update parameters
                    self.update_parameters(grads_w, grads_b, learning_rate)
                    
                    # Update progress bar
                    pbar.update(1)
                
                avg_loss = epoch_loss / num_samples
                #print(f"epoch_loss / num_samples: {epoch_loss:.1f}/{num_samples} = avg_loss: {avg_loss}")
                #print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
    
                # Early stopping based on average training loss
                if loss_threshold is not None and avg_loss < loss_threshold:
                    print(f"Early stopping at epoch {epoch+1} due to avg_loss < {loss_threshold}")
                    pbar.n = pbar.total  # Set progress bar to its maximum value
                    pbar.close()  # Close the progress bar
                    return  # Use return to exit the function instead of break
        
        print("Training complete.")
        pbar.close()
    
            
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        # Perform a forward pass through the network to compute activations for a single sample.
 
        activations = [x]
        current_active_layer = x
        for i in range(len(self.hidden_layers)):
            z = np.dot(current_active_layer, self.weights[i]) + self.biases[i]
            current_active_layer = self.relu(z)
            activations.append(current_active_layer)
        
        z = np.dot(current_active_layer, self.weights[-1]) + self.biases[-1]
        current_active_layer = self.softmax(z)
        activations.append(current_active_layer)
        
        return activations

    
    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        # Compute the categorical cross-entropy loss.
       
        # Adding a small value (1e-8) to y_pred to avoid taking the log of zero.
        loss = -np.sum(y * np.log(y_pred + 1e-8))
        return loss
    
    
    def compute_output_gradients(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Compute the gradient of the loss with respect to the output of the network (softmax).

        return y_pred - y
    
    
    def backward(self, activations: List[np.ndarray], y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Perform the backward pass to compute gradients for weights and biases.

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        
        # Compute gradient for the output layer
        delta = self.compute_output_gradients(y, activations[-1])
        grads_w[-1] = np.outer(activations[-2], delta)
        grads_b[-1] = delta
        
        # Backpropagate through hidden layers
        for i in range(len(self.hidden_layers)-1, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * (activations[i+1] > 0)  # Gradient of ReLU
            grads_w[i] = np.outer(activations[i], delta)
            grads_b[i] = delta
            
        return grads_w, grads_b
    
    
    def update_parameters(self, grads_w: List[np.ndarray], grads_b: List[np.ndarray], learning_rate: float) -> None:
        # Update weights and biases using computed gradients and learning rate.

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]
    
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Perform a forward pass through the network to make predictions.

        a = X
        for i in range(len(self.hidden_layers)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
        
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        return output

    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Calculate the accuracy of the neural network on the given dataset.
 
        # Perform predictions on the entire dataset
        y_pred = self.predict(X)

        # Calculate the number of correct predictions
        correct_predictions = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

        # Calculate accuracy
        accuracy = correct_predictions / X.shape[0]
        return accuracy


# In[3]:


def grid_search(X_train, y_train, X_val, y_val, hidden_layers_options, learning_rates, epochs_options, loss_threshold):
    best_accuracy = 0
    best_params = None

    # Sort epochs_options to ensure incremental training
    epochs_options = sorted(epochs_options)

    # Iterate over all combinations of hyperparameters
    for hidden_layers, learning_rate in product(hidden_layers_options, learning_rates):

        nn = NeuralNetwork(input_size=X_train.shape[1], output_size=num_classes, hidden_layers=hidden_layers)
        
        print(f"Testing with hidden_layers={hidden_layers}, {learning_rate=}, {epochs_options=}")
        
        for i, epochs in enumerate(epochs_options):
            additional_epochs = epochs if i == 0 else epochs - epochs_options[i-1]

            # Continue training the neural network
            nn.fit(X_train, y_train, epochs=additional_epochs, learning_rate=learning_rate, loss_threshold=loss_threshold)

            # Validate the neural network
            val_accuracy = nn.score(X_val, y_val)
            print(f"Validation Accuracy after {epochs} epochs: {val_accuracy * 100:.2f}%\n\n")

            # Update best parameters if current model is better
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = (hidden_layers, learning_rate, epochs)
    print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}% with parameters: {best_params}")
    return best_params, best_accuracy


# # MNIST

# ### preparing the data

# In[10]:


# Load the MNIST data
MNIST_train = pd.read_csv("csv files/MNIST-train.csv")

# Split the column named 'y' from the rest of the columns
y = MNIST_train['y']
X = MNIST_train.drop(columns=['y'])

# Normalize the input data
X = X / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_one_hot = np.zeros((y.size, num_classes))
y_one_hot[np.arange(y.size), y] = 1

# Use train_test_split to split the data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# ## A little warm-up to see that the model is actually working 

# In[11]:


# Create the neural network object
n = X_train.shape[1]
output_size = num_classes
hidden_layers = [10]

nn = NeuralNetwork(input_size=n, output_size=output_size, hidden_layers=hidden_layers)

# Train the neural network
nn.fit(X_train.values, y_train, epochs=1, learning_rate=0.01,loss_threshold=0.1)

# Test the neural network on the validation data
accuracy = nn.score(X_validation.values, y_validation)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# ## Let's find the best hyperparameters using a grid search!
# ### Beware! It took over 8 hours to complete...
# ### I copied the resaults to a file in named MNIST_grid for your convenience
# ### run this if you want to find a good set of parameters, once found there is no need to search again

# In[12]:


"""
seems like going from 784 layers to 10 is a bit of a strech
and yet we are getting a resault of arounf 90%
let's try a grid search to find the best hyperparameters
"""
#
## Define the hyperparameter grid
#hidden_layers_options = [
#    [64], [128], [150],[200],[250],
#    [64, 32], [128, 64], [200, 100],
#    [64, 64], [128, 128],[200, 50],
#]
#learning_rates = [0.05,0.01, 0.015, 0.02]
#epochs_options = [10, 20, 50]
#loss_threshold = 0.000001
#
#best_params, best_accuracy = grid_search(X_train.values, y_train, X_validation.values,
#                    y_validation, hidden_layers_options, learning_rates, epochs_options, loss_threshold)
#


# ##  We have found what looks to be good hyperparameters! 
# ### Now let's check the accuracy on the test set.
# ### But before we do, let's retrain the Neural Network using the validation set to make it even better!

# In[13]:


# Load the MNIST data
MNIST_train = pd.read_csv("csv files/MNIST-train.csv")
MNIST_test = pd.read_csv("csv files/MNIST-train.csv")

# Split the column named 'y' from the rest of the columns
y_train = MNIST_train['y']
X_train = MNIST_train.drop(columns=['y'])

y_test = MNIST_test['y']
X_test = MNIST_test.drop(columns=['y'])


# Convert labels to one-hot encoding
num_classes = 10
y_train_one_hot = np.zeros((y_train.size, num_classes))
y_train_one_hot[np.arange(y_train.size), y] = 1

y_test_one_hot = np.zeros((y_test.size, num_classes))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

# Normalize the input data
X_train = X_train / 255.0
X_test = X_test / 255.0

best_params = ([100],0.02,10) 
print(best_params)


# ## Let's see what resault we get with the best hyperparameters we found

# In[ ]:


# Train the best model on the entire training data (X and y_one_hot)
best_hidden_layers, best_learning_rate, best_epochs = best_params

best_nn = NeuralNetwork(input_size=X.shape[1], output_size=num_classes, hidden_layers=best_hidden_layers)
best_nn.fit(X_train.values, y_train_one_hot, epochs=best_epochs, learning_rate=best_learning_rate, loss_threshold=0.00001)

# Test the best model
test_accuracy = best_nn.score(X_test.values, y_test_one_hot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f'best parameters found: hidden layers: {best_hidden_layers}, learning rate: {best_learning_rate}, best epochs: {best_epochs}')


# # MB Train
# ## First we need to deal with the data

# In[ ]:


# Step 1: Load the Dataset
MB_train = pd.read_csv("MB_data_train.csv")

# Step 2: Preprocess the Data
# Extract labels
MB_train['label'] = MB_train.iloc[:, 0].apply(lambda x: 1 if 'Pt_Fibro' in x else 0)

# Drop the first column
MB_train.drop(MB_train.columns[0], axis=1, inplace=True)

# Separate features and labels
X = MB_train.drop('label', axis=1).values
y = MB_train['label'].values

# Normalize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert labels to one-hot encoding
num_classes = 2
y_one_hot = np.zeros((y.size, num_classes))
y_one_hot[np.arange(y.size), y] = 1

# Step 3: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)


# ## Now let's try a cupple of examples to narrow our search 

# In[ ]:


# Define the neural network architecture
input_size = X_train.shape[1]
output_size = num_classes
hidden_layers = [800]
# Initialize the neural network
nn = NeuralNetwork(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)

# Train the neural network
nn.fit(X_train, y_train, epochs=100, learning_rate=0.01, loss_threshold=0.00001)
# Evaluate the neural network on the validation set
accuracy = nn.score(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


# Define the neural network architecture
input_size = X_train.shape[1]
output_size = num_classes
hidden_layers = [500,250]
# Initialize the neural network
nn = NeuralNetwork(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)

# Train the neural network
nn.fit(X_train, y_train, epochs=75, learning_rate=0.02, loss_threshold=0.00001)
# Evaluate the neural network on the validation set
accuracy = nn.score(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


# Define the neural network architecture
input_size = X_train.shape[1]
output_size = num_classes
hidden_layers = [800,200,50]
# Initialize the neural network
nn = NeuralNetwork(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)

# Train the neural network
nn.fit(X_train, y_train, epochs=100, learning_rate=0.02, loss_threshold=0.00001)
# Evaluate the neural network on the validation set
accuracy = nn.score(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# ## It seems that hidden layers of [800,200,50] with 100 epochs and a learning rate of 0.02 works well and are (relatively) consistant , so lets use them :)

# #### Let's first train the model on the entire training set to get the best results 

# In[ ]:


# Load the Dataset
MB_train = pd.read_csv("MB_data_train.csv")

# Extract labels
MB_train['label'] = MB_train.iloc[:, 0].apply(lambda x: 1 if 'Pt_Fibro' in x else 0)

# Drop the first column
MB_train.drop(MB_train.columns[0], axis=1, inplace=True)

# Separate features and labels
X_train = MB_train.drop('label', axis=1).values
y_train = MB_train['label'].values

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Convert labels to one-hot encoding
num_classes = 2
y_train_one_hot = np.zeros((y_train.size, num_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1


# Define the neural network architecture
input_size = X_train.shape[1]
output_size = num_classes
hidden_layers = [800, 200, 50]

# Initialize the neural network
nn = NeuralNetwork(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)

# Train the neural network
nn.fit(X_train, y_train_one_hot, epochs=100, learning_rate=0.02, loss_threshold=0.00001)


# # Now let's use the actual test data set

# In[ ]:


# Load the Dataset
MB_test = pd.read_csv("../MB_data_test.csv")

# Extract labels
MB_test['label'] = MB_test.iloc[:, 0].apply(lambda x: 1 if 'Pt_Fibro' in x else 0)

# Drop the first column
MB_test.drop(MB_test.columns[0], axis=1, inplace=True)

# Separate features and labels
X_test = MB_test.drop('label', axis=1).values
y_test = MB_test['label'].values

# Normalize the input features
X_test = scaler.transform(X_test) 

# Convert labels to one-hot encoding
y_test_one_hot = np.zeros((y_test.size, num_classes))
y_test_one_hot[np.arange(y_test.size), y_test] = 1


# Evaluate the neural network on the test set
accuracy = nn.score(X_test, y_test_one_hot)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

