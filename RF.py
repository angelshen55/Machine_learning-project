# 安装必要库（如果未安装）
# pip install pandas openpyxl scikit-learn matplotlib
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
import joblib  # 模型保存

# 1. 数据读取与预处理
df = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')

# 确保包含Class列
if 'Class' not in df.columns:
    raise ValueError("数据中缺少Class列，请检查Excel文件结构")

# 处理缺失值
df = df.dropna()

# 2. 特征工程（保持原有特征处理）
# 添加您的特征工程代码（如果有）
# 例如：
# df['SF1_x_SF3'] = df['ShapeFactor1'] * df['ShapeFactor3']

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
    max_depth=8,
    min_samples_split=10,
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

# 混淆矩阵可视化
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=rf.classes_, 
            yticklabels=rf.classes_,
            cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 多分类ROC曲线
y_test_bin = label_binarize(y_test, classes=rf.classes_)
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算宏平均
fpr["macro"], tpr["macro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {rf.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.show()

# 6. 特征重要性分析
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[sorted_idx], y=X.columns[sorted_idx], palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 7. 模型保存与结果输出
# 保存模型
joblib.dump(rf, 'bean_classifier_rf.pkl')

# 保存预测结果
results_df = X_test.copy()
results_df['True_Class'] = y_test
results_df['Predicted_Class'] = y_pred
results_df.to_excel('RF_Predictions.xlsx', index=False)