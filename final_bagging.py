# bagging_model.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
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

def plot_decision_boundary(model, X, y, best_depth, title="Decision Boundary"):
    """生成并显示PCA降维后的决策边界"""
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 训练专用模型
    boundary_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=best_depth,
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
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    Z = boundary_model.predict(grid_pca)
    Z = Z.reshape(xx.shape)
    
    # 可视化
    plt.figure(figsize=(15, 12))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.tab10)
    
    # 绘制样本点
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
    
    plt.title(f"{title} (Standardized PCA)\nBest Depth: {best_depth}", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ================== 新增函数：绘制ROC曲线 ==================
def plot_roc_curve(model, X_test, y_test):
    """生成多类别ROC曲线和AUC值"""
    # 将标签二值化
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    n_classes = y_test_bin.shape[1]
    
    # 获取预测概率
    y_proba = model.predict_proba(X_test)
    
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算宏观平均
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 可视化设置
    plt.figure(figsize=(12, 10))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2'])
    
    # 绘制每个类别
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{model.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    
    # 绘制平均曲线
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multiclass ROC Curves', fontsize=16)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
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
    
    # 绘制决策边界
    plot_decision_boundary(best_model, X_train, y_train, best_depth)
    
    # 模型评估
    y_pred = bagging.predict(X_test)
    y_proba = bagging.predict_proba(X_test)  # 确保这行存在
    
    # 新增ROC曲线绘制
    plot_roc_curve(bagging, X_test, y_test)
    
    # 后续保持原有代码不变...



if __name__ == "__main__":
    main()