import torch
import torch.nn as nn
import torch.nn.functional as F

    
class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    




# class cifar10Net(nn.Module):
#     def __init__(self):
#         super(cifar10Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.fc1 = nn.Linear(32*4*4, 32*4*4)
#         self.fc2 = nn.Linear(32*4*4, 32*2*2)
#         self.fc3 = nn.Linear(32*2*2, 10)
#         self.dropout1 = nn.Dropout(0.4)  # 从默认值增加到0.4
#         self.dropout2 = nn.Dropout(0.6)  # 从默认值增加到0.6

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 32*4*4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# class cifar10Net(nn.Module):
#     def __init__(self):
#         super(cifar10Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.dropout = nn.Dropout(0.3)
#         # 注意特征图尺寸计算：64x4x4 = 1024
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = torch.flatten(x, 1)  # 批量维度不平展
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# class cifar10Net(nn.Module):
#     def __init__(self):
#         super(cifar10Net, self).__init__()
#         # 第一个卷积块
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         # 第二个卷积块
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         # 第三个卷积块 (减少池化次数)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
        
#         # 修正的全连接层 (只进行2次池化, 8x8x128=8192)
#         self.fc1 = nn.Linear(8192, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         # 第一个块 - 仅池化两次
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.pool(x)  # 32x32 -> 16x16
        
#         # 第二个块
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.pool(x)  # 16x16 -> 8x8
        
#         # 第三个块 - 不再继续池化
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
        
#         # 全连接层
#         x = torch.flatten(x, 1)  # 展平成 [batch_size, 8*8*128]
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
        
#         return x
# from resnet_v2 import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# def get_model(args, num_classes, model_input_channels):
#     if args.model.lower() == 'resnet10_v2':
#         return ResNet10(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
#     elif args.model.lower() == 'resnet18_v2':
#         return ResNet18(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
#     elif args.model.lower() == 'resnet34_v2':
#         return ResNet34(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
#     elif args.model.lower() == 'resnet50_v2':
#         return ResNet50(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
#     elif args.model.lower() == 'resnet101_v2':
#         return ResNet101(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
#     elif args.model.lower() == 'resnet152_v2':
#         return ResNet152(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
#     else:
#         raise ValueError(f"Unsupported model name: {args.model}")





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class cifar10Net(nn.Module):
    def __init__(self):
        super(cifar10Net, self).__init__()
        self.in_planes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差层
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # 32x32
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 16x16
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 8x8
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始层
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 全局平均池化和分类器
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class femnistNet(nn.Module):
    def __init__(self):
        super(femnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    
class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # SVHN has 3 color channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128) # Adjusted linear layer size
        self.fc2 = nn.Linear(128, 10)  # 10 classes for the digits 0-9

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # SVHN has 3 color channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjusted linear layer size
        self.fc2 = nn.Linear(128, 10)  # 10 classes for the digits 0-9

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_model(args, num_classes, model_input_channels):
    if args.dataset == 'FEMNIST':
        return femnistNet()
    elif args.dataset == 'CIFAR10':
        return cifar10Net()
    elif args.dataset == 'SVHN':
        return SVHNNet()
    elif args.dataset == 'MNIST':
        return mnistNet()
    else:
        # 默认根据通道数选择
        if model_input_channels == 1:
            return femnistNet()  # 单通道图像
        else:
            return cifar10Net()  # 多通道图像
    c
