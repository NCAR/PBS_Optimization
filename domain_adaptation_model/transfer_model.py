import torch
import torch.nn as nn
import math
import mmd
from functions import ReverseLayerF
#------------ Deep Adaptation Network (DAN) --------------#
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print ("Residual shape", residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        #print ("Residual, out", residual.shape, out.shape)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print ("After conv2-bn2-relu", out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        #print ("After conv3-bn3", out.shape)
        #print ("Residual", residual.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        #print ("After downsample", residual.shape)
        out += residual
        out = self.relu(out)

        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, config):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        #print ("Layer 1", self.layer1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        
        self.fc = nn.Linear(128, config['num_classes'])
        self.baselayer = [self.conv1,self.bn1,self.layer1,self.layer2,self.layer3,self.layer4]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print ("After conv1", out.shape)
        out = self.maxpool(out)
        #print ("Maxpool shape", out.shape)
        out = self.layer1(out)
        #print ("Layer 1 out", out.shape)
        out = self.layer2(out)
        #print ("Layer 2 out", out.shape)
        out = self.layer3(out)
        #print ("Layer 3", out.shape)
        out = self.avgpool(out)
        #print ("Avg pool", out.shape)
        out = out.view(out.size(0), -1)
        #print ("Reshape output", out.shape)
        #out = self.fc(out)
        return out

class DANNet(nn.Module):
        def __init__(self, config):
                super(DANNet, self).__init__()
                self.sharedNet = resnet50(config, False)
                self.cls_fc = nn.Linear(256, config['num_classes'])

        def forward(self, source, target):
                loss = 0
                source = self.sharedNet(source)
                #print ("Pass shared net")
                if (self.training == True):
                        target = self.sharedNet(target)
                        loss += mmd.mmd_rbf_noaccelerate(source, target)

                source = self.cls_fc(source)

                return source, loss

def resnet50(config, pretrained=False, **kwargs):
        """
        Construcs a ResNet-50 model
        """
        model = ResNet(Bottleneck, [3,4,6,3], config, **kwargs)
        return model


#----Unsupervised Domain Adaptation by Backpropagation------#
class CNNModel(nn.Module):
	def __init__(self, config):
		super(CNNModel, self).__init__()
		self.feature = nn.Sequential()
		self.feature.add_module('f_conv1', nn.Conv2d(1,64, kernel_size=5))
		self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
		self.feature.add_module('f_pool1', nn.MaxPool2d(2))
		self.feature.add_module('f_relu1', nn.ReLU(True))
		self.feature.add_module('f_conv2', nn.Conv2d(64,50, kernel_size=5))
		self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
		self.feature.add_module('f_drop1', nn.Dropout2d())
		self.feature.add_module('f_pool2', nn.MaxPool2d(2))
		self.feature.add_module('f_relu2', nn.ReLU(True))
		#print ("Feature module", self.feature)
		self.class_classifier= nn.Sequential()
		self.class_classifier.add_module('c_fc1', nn.Linear(50*13*13, 100))
		self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
		self.class_classifier.add_module('c_relu1', nn.ReLU(True))
		self.class_classifier.add_module('c_drop1', nn.Dropout2d())
		self.class_classifier.add_module('c_fc2', nn.Linear(100,100))
		self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
		self.class_classifier.add_module('c_relu2', nn.ReLU(True))
		self.class_classifier.add_module('c_fc3', nn.Linear(100,10))
		self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

		self.domain_classifier = nn.Sequential()
		self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 13 * 13, 100))
		self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
		self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
		self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
		self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
	def forward(self, input_data, alpha):
		#print ("Input data before", input_data.shape)
		input_data = input_data.expand(input_data.data.shape[0], 1, input_data.data.shape[3], input_data.data.shape[3])
		#print ("After expand", input_data.shape)
		feature = self.feature(input_data)
		#print ("After feature", feature.shape)
		feature = feature.view(-1,50*13*13)
		reverse_feature = ReverseLayerF.apply(feature, alpha)
		class_output = self.class_classifier(feature)
		domain_output = self.domain_classifier(reverse_feature)

		return class_output, domain_output

