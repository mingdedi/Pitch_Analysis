import torch
import torch.nn as nn
import torch.optim as optim

class CREPEModel(nn.Module):
    def __init__(self):
        super(CREPEModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=512, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=32, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 360),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#这个模型是我的一个设想，目的是从根源上解决模型窗口过长的问题
#我假设的样本长度为256，但是这个模型需要进一步验证
class Small_CREPEModel(nn.Module):
    def __init__(self):
        super(Small_CREPEModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=512, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=32, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)  
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 360),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    #加载模型实例
    model = CREPEModel()
    #模拟数据输入，注意输入数据格式
    input_tensor = torch.randn(32, 1, 1024)
    #测试模型输出
    output = model(input_tensor)

    #打印输出形状
    print("Output shape:", output.shape)
