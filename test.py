import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 假设你的数据是一个 10x10 的二维数组
data = np.random.rand(10, 10)

# 创建 x 轴和 y 轴的标签
x_labels = ['x' + str(i) for i in range(10)]
y_labels = ['y' + str(i) for i in range(10)]

# 创建热力图
sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels)

# 显示图形
plt.show()
plt.savefig('/root/02-ACD-Prompt_v1.0/heatmap.png')