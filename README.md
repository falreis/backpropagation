# Backpropagation and Gradient Descent
### Author: Jay Mody
---
This repo is a workspace for me to develop my own gradient descent algorithims implemented in python from scratch (no use of machine learning libraries except to generate data).


**Limitations and Features:**
So far, i've been able to succesfully claissify clusters of data (multiple classes) generated using make_blobs from SKLearn. It doesn't work everytime, with my testing I found here are a few reasons:

- Loss will increase possibly due to overshooting of gradients
- Sometimes the network will get stuck in a local minumum
- Since weights are initialized randomly, some initial configurations fail despite many epochs of training (no learning, local minimum)
- Activations hugely affected my results
- The netowkr suffers from vanishing gradient


As such, only about 70% of the time I find that the network was able to 'learn'. Although, I'm happy with the results, as this was more an excersize for me to understand backprop and gradient descent than actually implementing it for use. Here's an example:



![before](/imgs/before.png)


![training](/imgs/loss_acc.png)


![after](/imgs/after.png)



**Completed:**
- Created dataset
- Visualize data
- Implemented MLP class
- Feed Forward
- Backpropagation
- Training algorithim and update weights using GD with backprop
- Implemented SGD
- Implemented Mini-Batch GD
- Visualize outputs
- Test and Predict
- Generalize backprop and ff for any number of layers


**To Do:**
- Regularization
- Momentum
- Gradient Clipping
- Choose activations per layer