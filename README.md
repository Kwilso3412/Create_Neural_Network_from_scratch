# Create_Neural_network_from_scratch

## Why would you do this? Tensorflow and Pytorch already exist.
I wanted to get a deeper understanding of what is happening under the hood instead of just looking at the higher level concepts of machine learning and neural networks. 

## Alright what problem are you trying to solve?
Looking at the dataset I have three questions I want to learn:
  1.  What genre do the people in a country like to listen too
  2.  What are the total minutes streamed per platform of the users in a country
  3.  What are by age what are the preferred listening times of users (Morning/Afternoon/Night)

## What dataset did you use to train the data?
[Kaggle Dataset](https://www.kaggle.com/datasets/atharvasoundankar/global-music-streaming-trends-and-listener-insights?resource=download)

## Okay so tell more about the neural network.

The core methodology is identical for all three, but the mathematical details adapt to classification vs regression. 
- problem 1 and 3 are classication problems
- problem 2 is a regression problem

This shows proper neural network design - same framework, different loss functions and activations based on the problem type.

The neural network will have three layers:
- Oth layer:  input layer
- 1st layer:  hidden layer
- 2nd layer: output layer

Input Processing: 
- problem 1: One-hot
- problem 2: One-hot
- problem 3: Normalization

Output Processing:
- problem 1: One-hot
- problem 2: Scaling
- problem 3: One-hot


### TRAIN / VALIDATION / TEST METHODOLOGY:
1. Data Split: 60% train / 20% validation / 20% test
2. Training: Use train set to update weights
3. Validation: Monitor performance, early stopping
4. Testing: Final unbiased evaluation
5. Early Stopping: Prevents overfitting

### KEY EQUATIONS USED:

#### Classification:
- Output: softmax(z2) → probabilities
- Loss: -Σ(y_true * log(y_pred)) / m
- Forward propagation for classification:
  * z1 = X @ W1 + b1
  * a1 = ReLU(z1)
  * z2 = a1 @ W2 + b2
  * a2 = softmax(z2)  # Probability distribution
- Backprop: dz2 = a2 - y_true
- Metric: accuracy
  
#### Regression:
- Output: z2 → continuous values
- Forward propagation for regression:
  * z1 = X @ W1 + b1
  * a1 = ReLU(z1)
  * z2 = a1 @ W2 + b2  (Linear output, no activation)
- Loss: Σ(y_true - y_pred)² / (2m)
- Backprop: dz2 = (y_pred - y_true) / m
- Metric: R² score
