import torch
import torch.nn as nn

#CNN Model
# 2 convolutional layers
class ConvNet(nn.Module):
        def __init__(self, num_classes, device):
                super(ConvNet, self).__init__()
                self.layer1 = nn.Sequential(
                        #kernel size: size of filter,
                        # stride: rate at which kernel passes over input image
                        # padding: add layers of 0s to outside of image to make sure that the kernel properly passes over the edges of image
                        #output conv2d: optimal feaure map (representation of) input image
                        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.layer2 = nn.Sequential(
                        nn.Conv2d(16,32, kernel_size = 5, stride=1, padding=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.fc = nn.Linear(1*16*32, num_classes)
                self.device = device
        def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                return out

# Simple LSTM model
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
		super(RNN, self).__init__()
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True)
		self.hidden_size = hidden_size
		self.fc = nn.Linear(hidden_size,  num_classes)
		self.drop = nn.Dropout(0.8)
		self.device = device
	def forward(self, x):
		# Set intial hidden and cell states
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(self.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(self.device)
		#Forward LSTM
		out, _ = self.lstm(x, (h0, c0)) # dim (batch_size, seq_length, hidden_size)
		# Decode the hidden state of last time step
		out = self.fc(out[:, -1, :])
		return out


# Feed-forward Neural Network
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, device):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2  = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_classes)
		self.drop = nn.Dropout(0.5)
		self.device = device
	def forward(self,x):
		out = self.fc1(x)
		out =self.relu(out)
		out = self.fc2(out)
		out =self.relu(out)
		out = self.fc3(out)
		return out

