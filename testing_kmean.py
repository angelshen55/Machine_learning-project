# 安装必要库（在终端中执行）
# pip install pandas openpyxl scikit-learn matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 数据读取
df = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')

# 2. 数据预处理
# 删除非数值列（类别标签）
true_k = df['Class'].nunique()  # Class 列的唯一值数量
print(f"真实类别数 k = {true_k}")
if 'Class' in df.columns:
    df = df.drop('Class', axis=1)

# 处理缺失值（示例数据集似乎完整，此处保留以防万一）
df = df.dropna()

# 3. 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

best_k = true_k
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# 5. 可视化（使用PCA降维）
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

plt.figure(figsize=(12,8))
plt.scatter(principal_components[:,0], principal_components[:,1], 
            c=clusters, cmap='viridis', s=50, alpha=0.6)
plt.title('K-Means Clustering Results (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# 6. 评估（轮廓系数）
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"Silhouette Coefficient: {silhouette_avg:.3f}")

# 7. 保存聚类结果（可选）

df['Cluster'] = clusters
df.to_excel('Clustered_Beans.xlsx', index=False)