# bagging_model.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
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

# ================== 新增代码：决策边界可视化 ==================
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """生成并显示PCA降维后的决策边界"""
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 训练专用模型（保持相同参数）
    boundary_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=13,
            min_samples_split=10,
            random_state=RANDOM_STATE
        ),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.7,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    boundary_model.fit(X_pca, y_encoded)
    
    # 生成网格数据
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    Z = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 可视化设置
    plt.figure(figsize=(15, 12))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.tab10)
    
    # 绘制样本点（随机30%）
    np.random.seed(RANDOM_STATE)
    sample_mask = np.random.rand(len(X_pca)) < 0.3
    scatter = plt.scatter(
        X_pca[sample_mask, 0], 
        X_pca[sample_mask, 1], 
        c=y_encoded[sample_mask], 
        cmap=plt.cm.tab10,
        edgecolor='k',
        s=50
    )
    
    # 图例设置
    handles, _ = scatter.legend_elements()
    plt.legend(handles, le.classes_, 
              title="Bean Types", 
              bbox_to_anchor=(1.05, 1),
              loc='upper left')
    
    plt.title(f"{title} (PCA Projection)\nmax_depth={13}", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(alpha=0.3)
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
    
    # 在main函数最后添加调用
    plot_decision_boundary(best_model, X_train, y_train, "Bagging Decision Boundary")

if __name__ == "__main__":
    main()