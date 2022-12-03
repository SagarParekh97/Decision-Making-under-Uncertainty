
import torch
import torch.nn as nn


'''
Define a class that derives from the nn.Module from torch
'''
class model(nn.Module):
    
    def __init__(self):
        super(model, self).__init__()   ## This is tp initialize the parent class so we can use the methods from nn.Module

        '''
        Define a simple Linear NN
        
        There are other layers that you can use like convolutional, recurrent, etc. Just change Linear with the 
        kind of layer you want. You can look at the PyTorch documentation:  https://pytorch.org/docs/stable/nn.html
        The two arguments in parentheses are the input and output size of the layers.
        '''
        self.linear1 = nn.Linear(20, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

        '''
        You also need to define a loss function that is used to compare the prediction of the network with the actual 
        value. This can change based on the type of application. You can look at the available loss function here:
        https://pytorch.org/docs/stable/nn.html#loss-functions
        '''
        self.criterion = nn.BCELoss()

    '''
    You pass the input through the different layers of your model. You can also add acitvation functions to the output 
    of each layer. Documentation for the available activation functions: 
    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    '''
    def forward(self, x):
        y_hat = torch.relu(self.linear1(x))
        y_hat = torch.relu(self.linear2(y_hat))
        return torch.sigmoid(self.linear3(y_hat)).squeeze()

    def train(self, x, y):

        '''
        '''

        '''
        Forward function to predict the output from inputs
        '''
        y_hat = self.forward(x)

        '''
        Calculate the loss between the actual labels and the predicitons
        '''
        loss = self.criterion(y_hat, y)
        return loss


'''
Create an instance of the model class
'''
model_NN = model()

'''
Need an optimizer for gradient descent. More optimizers at:
https://pytorch.org/docs/stable/optim.html
'''
optimizer = torch.optim.Adam(model_NN.parameters(), lr=1e-4)
EPOCH = 500
batch_size = 20
LOSS = []

for epoch in range(EPOCH):
    '''
    x = Features or Inputs
    '''
    '''
    y = Labels
    '''

    '''
    Get the loss from the model. This is used for the optimizer step
    '''
    loss = model_NN.train(x, y)

    '''
    Taking a gradient step every epoch.
     - First clear the optimizer by using zero_grad().
     - Back propagate the loss with backward()
     - Take a gradient step with step()
    '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    LOSS.append(loss.item())
