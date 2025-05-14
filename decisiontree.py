import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 数据加载与校验
try:
    df = pd.read_excel(
        'Dry_Bean_Dataset.xlsx',
        skiprows=4,
        skipfooter=3,
        header=None,
        engine='openpyxl'  # 确保使用正确的解析引擎
    )
    
    # 手动指定列名（必须与Excel列顺序完全一致）
    expected_columns = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
        'Extent', 'Solidity', 'roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4',
        'Class'
    ]
    
    # 列数校验
    if df.shape[1] != len(expected_columns):
        raise ValueError(f"列数不匹配！预期 {len(expected_columns)} 列，实际 {df.shape[1]} 列")
    df.columns = expected_columns

    df['SF1_x_SF3'] = df['ShapeFactor1'] * df['ShapeFactor3']
    df['SF2_squared'] = df['ShapeFactor2'] ** 2
    df['SF4_log'] = np.log(df['ShapeFactor4'].abs() + 1e-6)  # 防止零值

    
    # 清洗数据
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='ignore')  # 确保Class列保持为字符串
    
    验证Class列
    if 'Class' not in df.columns:
        raise KeyError("数据中缺少Class列，请检查列名或文件结构")
    
    print("\n类别分布：")
    print(df['Class'].value_counts())

    drop_columns = [
    'ShapeFactor2', 'Compactness', 'Solidity', 'EquivDiameter',
    'Extent', 'ConvexArea', 'Area', 'Eccentricity', 'ShapeFactor4']
    df.drop(columns=drop_columns, inplace=True)

except Exception as e:
    print(f"数据加载失败: {str(e)}")
    exit()

# 划分数据集
X = df.drop('Class', axis=1)
y = df['Class']
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)


dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    random_state=42
)
dt.fit(X_train, y_train)

# 获取特征重要性
feature_importances = dt.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]  # 重要性从高到低排序

# 打印特征重要性
print("\n特征重要性排序：")
for idx in sorted_idx:
    print(f"{features[idx]:<15} {feature_importances[idx]:.4f}")

# 特征选择实验
best_accuracy = 0
best_features = []
results = []

# 从1个特征到全部特征逐步测试
for k in range(1, len(sorted_features := features[sorted_idx])+1):
    selected_features = sorted_features[:k]
    
    # 使用相同的数据划分
    X_train_sub = X_train[selected_features]
    X_test_sub = X_test[selected_features]
    
    # 重新训练模型
    dt_sub = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        random_state=42
    )
    dt_sub.fit(X_train_sub, y_train)
    
    # 评估模型
    y_pred_sub = dt_sub.predict(X_test_sub)
    accuracy = accuracy_score(y_test, y_pred_sub)
    results.append((k, accuracy, selected_features))
    
    print(f"使用前{k}个特征: 准确率 {accuracy:.2%}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features = selected_features

# 显示最佳结果
print(f"\n{' 最佳结果 ':=^40}")
print(f"最高准确率: {best_accuracy:.2%}")
print("使用特征:", best_features)

# 用最佳特征重新训练最终模型
final_dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    random_state=42
)
final_dt.fit(X_train[best_features], y_train)
y_pred_final = final_dt.predict(X_test[best_features])

print("\n最终模型验证:")
print(f"准确率: {accuracy_score(y_test, y_pred_final):.2%}")
print("分类报告:")
print(classification_report(y_test, y_pred_final))




    # print("\n分类报告:")
    # print(classification_report(y_test, y_pred))

# # 可视化
# plt.figure(figsize=(12, 8))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', 
#            xticklabels=dt.classes_,
#            yticklabels=dt.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # ROC曲线（多分类）
# y_test_bin = label_binarize(y_test, classes=dt.classes_)
# n_classes = y_test_bin.shape[1]

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'black'])
# plt.figure(figsize=(10, 8))

# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='{0} (AUC = {1:0.2f})'
#              ''.format(dt.classes_[i], roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Multiclass ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

# # 特征重要性
# feature_imp = pd.Series(dt.feature_importances_, 
#                        index=X.columns).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# sns.barplot(x=feature_imp, y=feature_imp.index)
# plt.title('Feature Importance')
# plt.show()