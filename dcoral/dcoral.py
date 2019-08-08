#import tensorflow as tf
#from tensorflow.contrib import slim
import torch
import torch.nn as nn
#class JDDA(nn.Module):
#	def __init__(self, config):
#		super(JDDA, self).__init__()
#		self.layer1 = nn.Sequential()
#		self.layer1.add_module('l1_conv1', nn.Conv1d(1,64,kernel_size=5))
#		self.layer1.add_module('l1_bn1', nn.BatchNorm1d(64))
#		self.layer1.add_module('l1_pool1', nn.MaxPool1d(2))
#		self.layer1.add_module('l1_relu1', nn.ReLU(True))
#		self.layer1.add_module('l1_conv2', nn.Conv1d(64,128, kernel_size=5))
#		self.layer1.add_module('l1_bn2', nn.BatchNorm1d(128))
#		self.layer1.add_module('l1_pool2', nn.MaxPool1d(2))
#		self.layer1.add_module('l1_relu2', nn.ReLU(True))
#		if (config['old'] == False):
#			size = 2048
#		else:
#			size = 128
#		self.fc1 = nn.Sequential()
#		self.fc1.add_module('fully_conected', nn.Linear(size, 1024))
#		self.fc1.add_module('drop', nn.Dropout(0.5))
#		self.fc2 = nn.Linear(1024, 6)
#		self.softmax = nn.Softmax(dim=0)
#	
#	def forward(self, x):
#		batch_size = x.shape[0]
#		out = self.layer1(x)
#		#print ("After layer 1", out.shape)
#		out = out.reshape(batch_size, -1)
#		out = self.fc1(out)
#		#print ("FC1 ", out.shape)
#		out = self.fc2(out)
#		#print ("FC2", out.shape)
#		hidden = out
#		out = self.softmax(out)
#		#print ("Softmax", out.shape)
#		return out, hidden

class JDDA(nn.Module):
	#def __init__(self,config):
	#	super(JDDA,self).__init__()
	#	self.layer1 = nn.Sequential()
	#	self.layer1.add_module('conv1', nn.Conv1d(1,96,kernel_size=5))
	#	self.layer1.add_module('bn1', nn.BatchNorm1d(96))
	#	self.layer1.add_module('pool1', nn.MaxPool1d(kernel_size=2))
	#	self.layer1.add_module('relu1', nn.ReLU(True))

	#	self.layer2 = nn.Sequential()
	#	self.layer2.add_module('conv2', nn.Conv1d(96,256, kernel_size=1))
	#	self.layer2.add_module('bn2', nn.BatchNorm1d(256))
	#	self.layer2.add_module('pool2', nn.MaxPool1d(kernel_size=2))
	#	self.layer2.add_module('relu2', nn.ReLU(True))
        #      
	#	self.layer3 = nn.Sequential()
	#	self.layer3.add_module('conv3', nn.Conv1d(256,384, kernel_size=1))
	#	self.layer3.add_module('bn3', nn.BatchNorm1d(384))
	#	self.layer3.add_module('pool3', nn.MaxPool1d(2))
	#	self.layer3.add_module('relu3', nn.ReLU(True))

	#	self.layer4 = nn.Sequential()
	#	self.layer4.add_module('conv4', nn.Conv1d(384,384, kernel_size=1))
	#	self.layer4.add_module('bn4', nn.BatchNorm1d(384))
	#	self.layer4.add_module('pool4', nn.MaxPool1d(2))
	#	self.layer4.add_module('relu4', nn.ReLU(True))

	#	self.layer5 = nn.Sequential()
	#	self.layer5.add_module('conv5', nn.Conv1d(384,256, kernel_size=1))
	#	self.layer5.add_module('bn5', nn.BatchNorm1d(256))
	#	self.layer5.add_module('pool5', nn.MaxPool1d(kernel_size=2))
	#	self.layer5.add_module('relu5', nn.ReLU(True))

	#	if (config['old'] == False):
	#		size = 256*2
	#	else:
	#		size = 256
	#	self.fc1 = nn.Sequential()
	#	self.fc1.add_module('fc1_linear', nn.Linear(size, 4096))
	#	self.fc1.add_module('fc1_relu', nn.ReLU(True))
        #      #self.fc1.add_module('drop', nn.Dropout(0.5))

	#	self.fc2 = nn.Sequential()
	#	self.fc2.add_module('fc2_linear', nn.Linear(4096, 4096))
	#	self.fc2.add_module('fc2_relu', nn.ReLU(True))
        #      #self.fc2 = nn.Linear(1024, 6)

	#	self.fc3 = nn.Sequential()
	#	self.fc3.add_module('fc3_linear', nn.Linear(4096, config['num_classes']))
	#	self.fc3.add_module('fc3_relu', nn.ReLU(True))

	#	self.softmax = nn.Softmax(dim=0)

	def __init__(self, config):
		super(JDDA, self).__init__()
		self.layer1 = nn.Sequential()
		self.layer1.add_module('conv1', nn.Conv1d(1,64,kernel_size=5))
		self.layer1.add_module('bn1', nn.BatchNorm1d(64))
		self.layer1.add_module('pool1', nn.MaxPool1d(2))
		self.layer1.add_module('relu1', nn.ReLU(True))

		self.layer2 = nn.Sequential()
		self.layer2.add_module('conv2', nn.Conv1d(64,128, kernel_size=1))
		self.layer2.add_module('bn2', nn.BatchNorm1d(128))
		self.layer2.add_module('pool2', nn.MaxPool1d(2))
		self.layer2.add_module('relu2', nn.ReLU(True))
                
		self.layer3 = nn.Sequential()
		self.layer3.add_module('conv3', nn.Conv1d(128,256, kernel_size=1))
		self.layer3.add_module('bn3', nn.BatchNorm1d(256))
		self.layer3.add_module('pool3', nn.MaxPool1d(2))
		self.layer3.add_module('relu3', nn.ReLU(True))

		self.layer4 = nn.Sequential()
		self.layer4.add_module('conv4', nn.Conv1d(256,512, kernel_size=1))
		self.layer4.add_module('bn4', nn.BatchNorm1d(512))
		self.layer4.add_module('pool4', nn.MaxPool1d(2))
		self.layer4.add_module('relu4', nn.ReLU(True))

		self.layer5 = nn.Sequential()
		self.layer5.add_module('conv5', nn.Conv1d(512,1024, kernel_size=1))
		self.layer5.add_module('bn5', nn.BatchNorm1d(1024))
		self.layer5.add_module('pool5', nn.MaxPool1d(2))
		self.layer5.add_module('relu5', nn.ReLU(True))

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
	
	def forward(self, x):
		batch_size = x.shape[0]
		# Feature Extractor
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = out.reshape(batch_size, -1)
		hidden = out

		# Classifier
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)
		
		out = self.softmax(out)
		return out, hidden
    
class JDDA_RNN(nn.Module):
	def __init__(self, input_size, config):
		super(JDDA_RNN, self).__init__()
		self.num_layers = config['num_layers']
		self.hidden_size = config['hidden_size']
		self.device = config['device']
		self.lstm = nn.LSTM(input_size, config['hidden_size'], config['num_layers'], batch_first=True)
		self.fc1 = nn.Sequential()
		self.fc1.add_module('fully_conected', nn.Linear(128, 1024))
		self.fc1.add_module('drop', nn.Dropout(0.5))
		self.fc2 = nn.Linear(1024, 6)
		self.softmax = nn.Softmax(dim=0)
	
	def forward(self,x):
		batch_size = x.shape[0]
		
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(self.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(self.device)

		out,_ = self.lstm(x, (h0,c0))
		out = self.fc1(out.reshape(batch_size, -1))
		hidden = self.fc2(out)
		out = self.softmax(hidden)
		return out, hidden

class JDDA_FF(nn.Module):
	def __init__(self, input_size, config):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, config['hidden_size'])
		self.relu = nn.ReLU()
		self.fc2  = nn.Linear(config['hidden_size'], config['hidden_size'])
		self.fc3 = nn.Linear(config['hidden_size'], config['num_classes'])
		self.drop = nn.Dropout(config['dropout'])
		self.softmax = nn.Softmax(dim=0)
	def forward(self,x):
		out = self.fc1(x)
		out =self.relu(out)
		out = self.fc2(out)
		out =self.relu(out)
		
		out = self.fc3(self.drop(out))
		hidden = out
		out = self.softmax(out)
		return out, hidden
	
