# bagging_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 配置参数
RANDOM_STATE = 42
TEST_SIZE = 0.3
FILE_PATH = 'DryBeanDataset/Dry_Bean_Dataset.xlsx'

# 数据加载与预处理
def load_data():
    df = pd.read_excel(
        FILE_PATH,
        skiprows=4,
        skipfooter=3,
        header=None,
        engine='openpyxl'
    )
    
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
    return df



# 主流程
def main():
    # 数据准备
    df = load_data()
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # 配置基学习器
    md = 5
    for md in range(5,30):
        base_tree = DecisionTreeClassifier(
            max_depth=md,
            min_samples_split=10,
            random_state=RANDOM_STATE
        )

        
        # 创建Bagging模型
        bagging = BaggingClassifier(
            estimator=base_tree,
            n_estimators=100,
            max_samples=0.8,
            max_features=0.7,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        # 训练模型
        bagging.fit(X_train, y_train)
        
        # 模型评估
        y_pred = bagging.predict(X_test)
        y_proba = bagging.predict_proba(X_test)
        print(f"max depth = {md} 准确率: {accuracy_score(y_test, y_pred):.2%}")


    # # 输出结果
    # print(f"准确率: {accuracy_score(y_test, y_pred):.2%}")
    # print("\n分类报告:")
    # print(classification_report(y_test, y_pred))
    
    # # 可视化混淆矩阵
    # plt.figure(figsize=(10, 8))
    # cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=bagging.classes_, 
    #             yticklabels=bagging.classes_)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()
    
    # # 多分类ROC曲线
    # y_test_bin = label_binarize(y_test, classes=bagging.classes_)
    # n_classes = y_test_bin.shape[1]
    
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    
    # # 宏平均ROC曲线
    # fpr["macro"], tpr["macro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # plt.figure(figsize=(10, 8))
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #              label=f'Class {bagging.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
    #          color='navy', linestyle=':', linewidth=4)
    
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Multi-class ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    
    # # 特征重要性（基于基学习器平均）
    # importances = np.mean([
    #     est.feature_importances_ for est in bagging.estimators_
    # ], axis=0)
    
    # sorted_idx = np.argsort(importances)[::-1]
    # print("\n特征重要性:")
    # for idx in sorted_idx:
    #     print(f"{X.columns[idx]:<15} {importances[idx]:.4f}")
    
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=importances[sorted_idx], y=X.columns[sorted_idx], palette="viridis")
    # plt.title("Feature Importances (Bagging)")
    # plt.xlabel("Importance Score")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
