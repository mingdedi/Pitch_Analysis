import pandas as pd
import matplotlib.pyplot as plt
#这个是用来画训练过程中损失的变化曲线的！！
# 读取CSV文件
data = pd.read_csv('./Code/train_validation.csv', header=None)

# 提取数据
epochs = data[0]
train_loss = data[1]
validation_loss = data[2]

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', color='red')

# 设置图像标题和标签
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 显示图像
plt.show()
