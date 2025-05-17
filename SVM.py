import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


# 1. 数据读取与预处理
df = pd.read_excel('DryBeanDataset/Dry_Bean_Dataset.xlsx', engine='openpyxl')

# 确保包含Class列
if 'Class' not in df.columns:
    raise ValueError("数据中缺少Class列，请检查Excel文件结构")

# 处理缺失值
df = df.dropna()

# 2. 特征工程（基于最佳特征组合）
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

# 4. 使用最佳参数构建模型
best_params = {
    'C': 10,
    'gamma': 0.1,
    'kernel': 'rbf',
    'class_weight': None,
    'probability': True,
    'random_state': 42
}

# 创建带标准化的流水线
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(**best_params))
])

pipe.fit(X_train, y_train)

# 5. 模型评估
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)

# 输出指标
print("="*60)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("="*60)

# 混淆矩阵
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=pipe.classes_, 
            yticklabels=pipe.classes_,
            cmap='Blues')
plt.title('Optimized SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 多分类ROC曲线
y_test_bin = label_binarize(y_test, classes=pipe.classes_)
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 宏平均曲线
fpr["macro"], tpr["macro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10, 8))
colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{pipe.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4,
         alpha=0.8)

plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Optimized SVM ROC Curves', fontsize=14)
plt.legend(loc="lower right", frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================== 新增代码：准确率计算和混淆矩阵生成 ==================
import os
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_and_plot(model, X, y_true, dataset_name):
    """统一评估模型并保存混淆矩阵"""
    # 创建保存目录
    os.makedirs("results/confusion_matrices", exist_ok=True)
    
    # 预测结果
    y_pred = model.predict(X)
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{dataset_name}准确率: {accuracy:.2%}")
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 可视化设置
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap='Blues',
                annot_kws={'size': 8})
    
    plt.title(f'Confusion Matrix ({dataset_name})', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # 保存图片
    filename = f"results/confusion_matrices/{dataset_name.lower().replace(' ', '_')}_confusion.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {filename}")

# ================== 执行评估 ==================
print("\n" + "="*60)
print("模型综合评估结果")
print("="*60)

# 1. 训练集评估
evaluate_and_plot(pipe, X_train, y_train, "Training Set")

# 2. 测试集评估 (替换原有可视化代码)
evaluate_and_plot(pipe, X_test, y_test, "Testing Set")

# 3. 完整数据集评估
full_X = X  # 使用原始特征数据（确保与训练时特征一致）
full_y = y
evaluate_and_plot(pipe, full_X, full_y, "Full Dataset")

print("\n" + "="*60)


# ================== 新增：定义可视化所需特征 ==================
# 使用所有原始特征（不改变现有特征工程）
best_features = X.columns.tolist()  # 获取全部特征列名

# ================== 修改决策边界可视化调用 ==================
print("\n生成决策边界可视化...")
plot_svm_decision_boundary(pipe, X_train[best_features], y_train)

# ================== SVM决策边界可视化 ==================
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

def plot_svm_decision_boundary(model, X, y, title='SVM Decision Boundary'):
    """生成带决策边界的二维可视化"""
    # 使用PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(model.named_steps['scaler'].transform(X))
    
    # 在降维空间重新训练简化模型（保持相同参数）
    svm_pca = SVC(**model.named_steps['svm'].get_params())
    svm_pca.fit(X_pca, y)
    
    # 生成网格数据
    h = 0.1  # 网格步长
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点（使用降维空间模型）
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 可视化设置
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF', 
                                                          '#FFFF00', '#FF00FF', '#00FFFF', 
                                                          '#800080']))
    
    # 绘制原始数据点（抽样50%加速渲染）
    np.random.seed(42)
    sample_mask = np.random.rand(len(X_pca)) < 0.5
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        idx = (y == cls) & sample_mask
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                    c=[plt.cm.tab10(i)], 
                    edgecolor='black',
                    label=cls,
                    s=50,
                    alpha=0.9)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'{title} (PCA Projection)', fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 执行可视化（使用最佳特征）
print("\n生成决策边界可视化...")
plot_svm_decision_boundary(pipe, X_train[best_features], y_train)