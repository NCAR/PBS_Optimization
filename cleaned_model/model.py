import torch
import torch.nn as nn
import math
import mmd

#CNN Model
# 2 convolutional layers

#class ConvNet(nn.Module):
#        def __init__(self, config):
#                super(ConvNet, self).__init__()
#                self.layer1 = nn.Sequential(
#                        #kernel size: size of filter,
#                        # stride: rate at which kernel passes over input image
#                        # padding: add layers of 0s to outside of image to make sure that the kernel properly passes over the edges of image
#                        #output conv2d: optimal feaure map (representation of) input image
#                        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#                        nn.BatchNorm2d(16),
#                        nn.ReLU(),
#                       nn.MaxPool2d(kernel_size=2, stride=2)
#                )
#                self.layer2 = nn.Sequential(
#                        nn.Conv2d(16,32, kernel_size = 5, stride=1, padding=2),
#                        nn.BatchNorm2d(32),
#                        nn.ReLU(),
#                        nn.MaxPool2d(kernel_size=2, stride=2)
#                )
#                if (config['old'] == False):
#                        cur_size = 1 * 19 * 32
#                else:
#                        cur_size = 1 * 4 * 32
#                self.fc = nn.Linear(cur_size, config['num_classes'])
#                self.device = config['device']
#                self.drop = nn.Dropout(config['dropout'])
#        def forward(self, x):
#                #print ("Input shape", x.shape)
#                out = self.layer1(x)
#                #print ("Out layer 1", out.shape)
#                out = self.layer2(out)
#                #print ("Out layer 2", out.shape)
#                out = out.reshape(out.size(0), -1)
#                out = self.fc(self.drop(out))
#                return out


# AlexNet architecture for Convolutional Network 
class ConvNet(nn.Module):
        def __init__(self, config):
                super(ConvNet, self).__init__()
                self.layer1 = nn.Sequential()
                self.layer1.add_module('conv1', nn.Conv1d(1,64,kernel_size=5))
                self.layer1.add_module('bn1', nn.BatchNorm1d(64))
                self.layer1.add_module('pool1', nn.MaxPool1d(2))
                self.layer1.add_module('relu1', nn.LeakyReLU(True))
                
                self.layer2 = nn.Sequential()
                self.layer2.add_module('conv2', nn.Conv1d(64,128, kernel_size=1))
                self.layer2.add_module('bn2', nn.BatchNorm1d(128))
                self.layer2.add_module('pool2', nn.MaxPool1d(2))
                self.layer2.add_module('relu2', nn.LeakyReLU(True))

                self.layer3 = nn.Sequential()
                self.layer3.add_module('conv3', nn.Conv1d(128,256, kernel_size=1))
                self.layer3.add_module('bn3', nn.BatchNorm1d(256))
                self.layer3.add_module('pool3', nn.MaxPool1d(2))
                self.layer3.add_module('relu3', nn.LeakyReLU(True))

                self.layer4 = nn.Sequential()
                self.layer4.add_module('conv4', nn.Conv1d(256,512, kernel_size=1))
                self.layer4.add_module('bn4', nn.BatchNorm1d(512))
                self.layer4.add_module('pool4', nn.MaxPool1d(2))
                self.layer4.add_module('relu4', nn.LeakyReLU(True))

                self.layer5 = nn.Sequential()
                self.layer5.add_module('conv5', nn.Conv1d(512,1024, kernel_size=1))
                self.layer5.add_module('bn5', nn.BatchNorm1d(1024))
                self.layer5.add_module('pool5', nn.MaxPool1d(2))
                self.layer5.add_module('relu5', nn.LeakyReLU(True))
                #size=2048		
                if (config['old'] == False):
                        size = 2048
                else:
                        size = 1024
                self.fc1 = nn.Sequential()
                self.fc1.add_module('fc1_linear', nn.Linear(size, 2048))
                self.fc1.add_module('fc1_relu', nn.ReLU(True))
                #self.fc1.add_module('drop', nn.Dropout(0.5))

                self.fc2 = nn.Sequential()
                self.fc2.add_module('fc2_linear', nn.Linear(2048, 4096))
                self.fc2.add_module('fc2_relu', nn.ReLU(True))
                #self.fc2 = nn.Linear(1024, 6)

                self.fc3 = nn.Sequential()
                self.fc3.add_module('fc3_linear', nn.Linear(4096, config['num_classes']))
                self.fc3.add_module('fc3_relu', nn.ReLU(True))

                self.softmax = nn.Softmax(dim=0)
                self.drop = nn.Dropout(config['dropout'])
        def forward(self, x):
                batch_size = x.shape[0]

		# Feature Extractor
                out = self.layer1(x)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.layer5(out)
                out = out.reshape(batch_size, -1)

		# Classifier
                out = self.fc1(out)
                out = self.fc2(out)
                out = self.fc3(self.drop(out))
                out = self.softmax(out)
                return out

# Simple Bi-LSTM model
class RNN(nn.Module):
	def __init__(self, input_size, config):
		super(RNN, self).__init__()
		self.num_layers = config['num_layers']
		self.hidden_size = config['hidden_size']
		self.lstm = nn.LSTM(input_size, config['hidden_size'], config['num_layers'], batch_first= True, bidirectional=True)
		self.fc = nn.Linear(config['hidden_size'] * 2,  config['num_classes'])
		self.drop = nn.Dropout(config['dropout'])
		self.device = config['device']
	def forward(self, x):
		# Set intial hidden and cell states
		h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).float().to(self.device)
		c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).float().to(self.device)
		#Forward LSTM
		out, _ = self.lstm(x, (h0, c0)) # dim (batch_size, seq_length, hidden_size)
		# Decode the hidden state of last time step
		out = self.fc(self.drop(out[:, -1, :]))
		return out


# Simple Feed-forward Neural Network
class NeuralNet(nn.Module):
	def __init__(self, input_size, config):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, config['hidden_size'])
		self.relu = nn.ReLU()
		self.fc2  = nn.Linear(config['hidden_size'], config['hidden_size'])
		self.fc3 = nn.Linear(config['hidden_size'], config['num_classes'])
		self.drop = nn.Dropout(config['dropout'])
		self.device = config['device']
	def forward(self,x):
		out = self.fc1(x)
		out =self.relu(out)
		out = self.fc2(out)
		out =self.relu(out)
		out = self.fc3(self.drop(out))
		return out

###----- ResNet Model ---------- #####
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
#class ResNet(nn.Module):
#    def __init__(self, block, layers, config):
#        super(ResNet, self).__init__()
#        self.in_channels = 16
#        self.conv = conv3x3(1, 16)
#        self.bn = nn.BatchNorm2d(16)
#        self.relu = nn.ReLU(inplace=True)
#        self.layer1 = self.make_layer(block, 16, layers[0])
#        self.layer2 = self.make_layer(block, 32, layers[1], 2)
#        self.layer3 = self.make_layer(block, 64, layers[2], 2)
#        self.avg_pool = nn.AvgPool2d(8)
#        #print ("Avg pool result", self.avg_pool)
#        self.fc = nn.Linear(128, config['num_classes'])
#        self.drop = nn.Dropout(config['dropout'])
#
#    def make_layer(self, block, out_channels, blocks, stride=1):
#        downsample = None
#        if (stride != 1) or (self.in_channels != out_channels):
#            downsample = nn.Sequential(
#                conv3x3(self.in_channels, out_channels, stride=stride),
#
#        layers = []
#        layers.append(block(self.in_channels, out_channels, stride, downsample))
#        self.in_channels = out_channels
#        for i in range(1, blocks):
#            layers.append(block(out_channels, out_channels))
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        out = self.conv(x)
#        out = self.bn(out)
#        out = self.relu(out)
#        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        #print ("Layer 3", out.shape)
#        out = self.avg_pool(out)
#        out = out.view(out.size(0), -1)
#        #print ("Reshape output", out.shape)
#        out = self.fc(self.drop(out))
#        return out


		
