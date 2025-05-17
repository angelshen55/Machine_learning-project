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

def evaluate_model(model, X_train, y_train, X_test, y_test, X_full, y_full):
    """计算并打印三个数据集的准确率"""
    # 训练集评估
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # 测试集评估
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # 完整数据集评估
    y_full_pred = model.predict(X_full)
    full_acc = accuracy_score(y_full, y_full_pred)
    
    return train_acc, test_acc, full_acc

def plot_confusion_matrices(model, X_train, y_train, X_test, y_test, X_full, y_full, classes):
    """生成并显示三个数据集的混淆矩阵"""
    datasets = [
        (X_train, y_train, "Training Set"),
        (X_test, y_test, "Testing Set"),
        (X_full, y_full, "Full Dataset")
    ]
    
    for X, y, title in datasets:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=classes, 
                    yticklabels=classes,
                    cmap='Blues',
                    annot_kws={'size': 8})
        plt.title(f'Confusion Matrix ({title})', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

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
    
    # 跟踪最佳参数
    best_depth = 5
    best_accuracy = 0.0
    best_model = None
    
    # 深度调优
    for md in range(5, 20):
        base_tree = DecisionTreeClassifier(
            max_depth=md,
            min_samples_split=10,
            random_state=RANDOM_STATE
        )
        
        bagging = BaggingClassifier(
            estimator=base_tree,
            n_estimators=100,
            max_samples=0.8,
            max_features=0.7,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"max_depth={md} 准确率: {acc:.2%}")
        
        # 更新最佳模型
        if acc > best_accuracy:
            best_accuracy = acc
            best_depth = md
            best_model = bagging

    # 最终验证
    train_acc, test_acc, full_acc = evaluate_model(
        best_model,
        X_train, y_train,
        X_test, y_test,
        X, y
    )
    
    print(f"\n{' 最佳结果 ':=^40}")
    print(f"最佳max_depth: {best_depth}")
    print(f"训练集准确率: {train_acc:.2%}")
    print(f"测试集准确率: {test_acc:.2%}") 
    print(f"完整数据集准确率: {full_acc:.2%}")
    print("="*40)

    # 生成混淆矩阵
    plot_confusion_matrices(best_model, 
                           X_train, y_train,
                           X_test, y_test,
                           X, y,
                           classes=best_model.classes_)

if __name__ == "__main__":
    main()