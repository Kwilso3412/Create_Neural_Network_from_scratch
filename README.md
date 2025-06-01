# Create_Neural_network_from_scratch

## Why would you do this? Tensorflow and Pytorch already exist.
I wanted to get a deeper understanding of what is happening under the hood instead of just looking at the higher level concepts of machine learning and neural networks. 

## Alright what problem are you trying to solve?
Looking at the dataset I have three questions I want to learn:
  - What genre do the people in a country like to listen too
  - What are the total minutes streamed per platform of the users in a country
  - What are by age what are the preferred listening times of users (Morning/Afternoon/Night)

## What dataset did you use to train the data?
[Kaggle Dataset](https://www.kaggle.com/datasets/atharvasoundankar/global-music-streaming-trends-and-listener-insights?resource=download)

## Okay so tell more about the neural network.
The neural network will have three layers
- Oth layer:  input layer
- 1st layer:  hidden layer
- 2nd layer: output layer

### TRAIN/VALIDATION/TEST METHODOLOGY:
1. Data Split: 60% train / 20% validation / 20% test
2. Training: Use train set to update weights
3. Validation: Monitor performance, early stopping
4. Testing: Final unbiased evaluation
5. Early Stopping: Prevents overfitting

### KEY EQUATIONS USED:

1. Forward Pass:
  - z1 = X @ W1 + b1
  - a1 = ReLU(z1) = max(0, z1)
  - z2 = a1 @ W2 + b2
  - a2 = softmax(z2)
2. Loss Function:
  - L = -Σ(y_true * log(y_pred)) / m
3. Backpropagation:
  - dL/dW2 = a1^T @ (a2 - y_true) / m
  - dL/dW1 = X^T @ (dL/da1 * ReLU'(z1)) / m
4. Weight Updates:
  - W = W - learning_rate * dL/dW
5. Early Stopping:
  - Stop training when validation loss stops improving
