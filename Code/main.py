import torch
import torch.nn as nn
import torch.optim as optim
from model import CREPEModel
from voiceDataLoader import MIR_1K_Dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
#这个文件是用来训练模型的
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#He初始化
def he_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

#加载训练集和验证集
Vali_dataset = MIR_1K_Dataset('./Data/MIR-1K/Validation')
Vali_dataloader = DataLoader(Vali_dataset, batch_size=32, shuffle=True)
Train_dataset = MIR_1K_Dataset('./Data/MIR-1K/Train')
Train_dataloader = DataLoader(Train_dataset, batch_size=32, shuffle=True)
#初始化模型、损失函数、优化器
model = CREPEModel().to(device)
#model.apply(he_init)  # Apply He initialization
model.load_state_dict(torch.load('./CREPE.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

num_epochs = 50
#这个csv文件用来储存训练中的训练集损失和验证机损失的变化的
csv_file=open('train_validation.csv', mode='w', newline='')
for epoch in range(num_epochs):
    #每一轮的训练都要把总损失清零
    model.train()
    train_total_loss = 0.0
    val_total_loss = 0.0
    #训练循环
    for inputs, targets in Train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
    #打印训练平均损失
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_total_loss/len(Train_dataloader)},",end="")

    #记得开启评估模式
    model.eval()
    with torch.no_grad():
        #验证循环
        for val_inputs, val_targets in Vali_dataloader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            
            #累加损失和样本数
            val_total_loss += val_loss.item() * val_inputs.size(0)
        print(f"Validation Loss: {val_total_loss/3199}")
    #在训练完成和验证完成后，将损失写入到csv文件中，方便后续进行统计 
    writer = csv.writer(csv_file)
    writer.writerow([epoch+1,train_total_loss/len(Train_dataloader),val_total_loss/3199])
#训练完成后关系csv_file文件，并且保存模型权重
csv_file.close()
torch.save(model.state_dict(), './CREPE.pth')
