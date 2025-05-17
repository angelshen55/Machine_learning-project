import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from itertools import cycle

# ================== 1. 数据加载与预处理 ==================
try:
    # 加载数据
    df = pd.read_excel(
        'DryBeanDataset/Dry_Bean_Dataset.xlsx',
        skiprows=4,
        skipfooter=3,
        header=None,
        engine='openpyxl'
    )
    
    # 列名校验
    expected_columns = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
        'Extent', 'Solidity', 'roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4',
        'Class'
    ]
    
    if df.shape[1] != len(expected_columns):
        raise ValueError(f"列数不匹配！预期 {len(expected_columns)} 列，实际 {df.shape[1]} 列")
    df.columns = expected_columns

    # 特征工程
    df['SF1_x_SF3'] = df['ShapeFactor1'] * df['ShapeFactor3']
    df['SF2_squared'] = df['ShapeFactor2'] ** 2
    df['SF4_log'] = np.log(df['ShapeFactor4'].abs() + 1e-6)

    # 清洗数据
    df = df.dropna().reset_index(drop=True)
    
    # 标签编码
    le = LabelEncoder()
    df['Class_encoded'] = le.fit_transform(df['Class'])  # 新增编码列
    
    print("\n类别分布：")
    print(df['Class'].value_counts())

except Exception as e:
    print(f"数据加载失败: {str(e)}")
    exit()

# ================== 2. 数据划分 ==================
X = df.drop(['Class', 'Class_encoded'], axis=1)
y = df['Class_encoded']  # 使用编码后的标签

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y  # 分层抽样需要数值标签
)

# ================== 3. 决策树训练 ==================
dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    random_state=42
)
dt.fit(X_train, y_train)

# ================== 4. 特征重要性分析 ==================
feature_importances = dt.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]

# 可视化
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[sorted_idx], y=features[sorted_idx], palette="viridis")
plt.title("Decision Tree Feature Importances", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.show()

# ================== 5. 特征选择实验 ==================
best_accuracy = 0
best_features = []
print("\n特征选择实验:")

for k in range(1, len(features)+1):
    selected_features = features[sorted_idx][:k].tolist()
    
    dt_sub = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        random_state=42
    )
    dt_sub.fit(X_train[selected_features], y_train)
    
    y_pred = dt_sub.predict(X_test[selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"使用前{k}个特征 ({', '.join(selected_features)}) : 准确率 {accuracy:.2%}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features = selected_features

print(f"\n{' 最佳结果 ':=^40}")
print(f"最高准确率: {best_accuracy:.2%}")
print("最优特征组合:", best_features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from itertools import cycle


# ================== 6. 最终模型验证 ==================
final_dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    random_state=42
)
final_dt.fit(X_train[best_features], y_train)

# 1. 训练集准确率
y_train_pred = final_dt.predict(X_train[best_features])
train_accuracy = accuracy_score(y_train, y_train_pred)

# 2. 测试集准确率 (已存在)
y_test_pred = final_dt.predict(X_test[best_features])
test_accuracy = accuracy_score(y_test, y_test_pred)

# 3. 完整数据集准确率 (新增)
X_all = df.drop(['Class', 'Class_encoded'], axis=1)[best_features]
y_all = df['Class_encoded']
y_all_pred = final_dt.predict(X_all)
all_dataset_accuracy = accuracy_score(y_all, y_all_pred)

print("\n最终模型验证:")
print(f"训练集准确率: {train_accuracy:.2%}")
print(f"测试集准确率: {test_accuracy:.2%}")
print(f"完整数据集准确率: {all_dataset_accuracy:.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))


# ================== 7. Confusion Matrix Generation ==================
def plot_confusion_matrix(y_true, y_pred, title):
    """生成并立即显示混淆矩阵"""
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_,
                cmap='Blues',
                annot_kws={'size': 10})
    
    plt.title(f'Confusion Matrix ({title})', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()  # 立即显示当前图形

# ================== 调用部分修改 ==================
# 1. Training Set Evaluation
y_train_pred = final_dt.predict(X_train[best_features])
plot_confusion_matrix(y_train, y_train_pred, 'Training Set')

# 2. Testing Set Evaluation 
plot_confusion_matrix(y_test, y_test_pred, 'Testing Set')

# 3. Full Dataset Evaluation
X_all = df.drop(['Class', 'Class_encoded'], axis=1)[best_features]
y_all = df['Class_encoded']
y_all_pred = final_dt.predict(X_all)
plot_confusion_matrix(y_all, y_all_pred, 'Full Dataset')

# ================== 8. 决策边界可视化 (修复版) ==================
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# 1. PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train[best_features])

# 2. 转换为NumPy数组避免索引问题
pc1 = X_pca[:, 0]  # 第一主成分
pc2 = X_pca[:, 1]  # 第二主成分

# 3. 训练专用模型
dt_pca = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    random_state=42
)
dt_pca.fit(X_pca, y_train)  # 直接使用NumPy数组

# 4. 生成网格数据
x_min, x_max = pc1.min()-1, pc1.max()+1
y_min, y_max = pc2.min()-1, pc2.max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 5. 预测网格点（使用NumPy数组）
grid_pred = dt_pca.predict(np.c_[xx.ravel(), yy.ravel()])
grid_pred = grid_pred.reshape(xx.shape)

# 6. 可视化设置
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800080']
cmap = ListedColormap(colors[:len(le.classes_)])
markers = ['o', 's', '^', 'v', 'D', 'p', '*']

plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, grid_pred, alpha=0.4, cmap=cmap)

# 7. 绘制样本点（基于数组索引）
np.random.seed(42)
sample_mask = np.random.rand(len(X_pca)) < 0.3  # 正确长度

for i, class_name in enumerate(le.classes_):
    # 获取当前类别的编码
    class_code = le.transform([class_name])[0]
    
    # 生成类别掩码（数组操作）
    class_mask = (y_train.values == class_code)
    
    # 组合掩码
    mask = class_mask & sample_mask
    
    plt.scatter(pc1[mask], pc2[mask],
                c=colors[i],
                marker=markers[i],
                label=class_name,
                edgecolor='black',
                s=50)

plt.title("Decision Boundaries (PCA Projection)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

