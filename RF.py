import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle 

# 1. 数据读取与预处理
df = pd.read_excel('DryBeanDataset/Dry_Bean_Dataset.xlsx', engine='openpyxl')

# 确保包含Class列
if 'Class' not in df.columns:
    raise ValueError("数据中缺少Class列，请检查Excel文件结构")

# 处理缺失值
df = df.dropna()

# 2. 特征工程（保持原有特征处理）
df['SF1_x_SF3'] = df['ShapeFactor1'] * df['ShapeFactor3']
df['SF2_squared'] = df['ShapeFactor2'] ** 2
df['SF4_log'] = np.log(df['ShapeFactor4'].abs() + 1e-6)

# 3. 数据准备
X = df.drop('Class', axis=1)
y = df['Class']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    stratify=y,
    random_state=42
)

# 4. 模型训练
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    max_features=0.8,
    class_weight='balanced',  # 处理类别不平衡
    n_jobs=-1,  # 使用所有CPU核心
    random_state=42
)

rf.fit(X_train, y_train)

# 5. 模型评估
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

# 基础指标
print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))



# ================== 新增代码：模型综合评估 ==================
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, X, y_true, dataset_name):
    """统一评估模型并显示结果"""
    # 预测结果
    y_pred = model.predict(X)
    
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"{dataset_name}准确率: {acc:.2%}")
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 可视化
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_,
                cmap='Blues',
                annot_kws={'size': 10})
    plt.title(f'Confusion Matrix ({dataset_name})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

# 执行评估
print("\n" + "="*60)
print("随机森林模型综合评估")
print("="*60)

# 1. 训练集评估
evaluate_model(rf, X_train, y_train, "Training Set")

# 2. 测试集评估
evaluate_model(rf, X_test, y_test, "Testing Set")

# 3. 完整数据集评估
evaluate_model(rf, X, y, "Full Dataset")
print("="*60 + "\n")

# ================== 决策边界可视化 ==================
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap

# 1. 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 统一编码所有标签

# 2. PCA降维（使用完整数据集）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # X已包含所有特征（包括工程特征）

# 3. 训练专用RF模型（保持相同参数）
rf_pca = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    max_features=0.8,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_pca.fit(X_pca, y_encoded)  # 使用编码后的标签

# 4. 生成网格数据
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),  # 分辨率调整为0.1
                     np.linspace(y_min, y_max, 200))

# 5. 预测网格点（数值预测结果）
Z = rf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 6. 可视化设置
plt.figure(figsize=(15, 12))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#AAAAFF','#FFFFAA','#FFAAFF','#AAFFFF','#FFD700'])
plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

# 7. 绘制样本点（随机采样30%）
np.random.seed(42)
sample_mask = np.random.rand(len(X_pca)) < 0.3
scatter = plt.scatter(
    X_pca[sample_mask, 0], 
    X_pca[sample_mask, 1], 
    c=y_encoded[sample_mask], 
    cmap=ListedColormap(['#FF0000', '#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#FFA500']),
    edgecolor='k',
    s=50,
    label=le.classes_
)

# 8. 自定义图例
handles, _ = scatter.legend_elements()
plt.legend(handles, le.classes_, 
          title="Bean Types", 
          loc='upper left',
          bbox_to_anchor=(1.02, 1))

plt.title("Random Forest Decision Boundaries (PCA Projection)", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()