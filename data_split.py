# import pandas as pd
# from sklearn.model_selection import train_test_split
# from utils.utils import seed_everything
#
# Seed = 42
#
# # 固定随机种子
# seed_everything(seed=Seed)
#
# # 读取CSV文件
# df_spectral = pd.read_csv('data/2015/process/Spe_OC_N_CaCO3_pH.csv')
#
# # 随机分割数据集为训练集和测试集
# train, validation_test = train_test_split(df_spectral, train_size=0.7, random_state=Seed)
# validation, test = train_test_split(validation_test, train_size=0.5, random_state=Seed)
#
# # 保存 DataFrame 为 CSV 文件，同时指定 index=False 以不保存索引
# train.to_csv('data/2015/process/Train_Set.csv', index=False)
# validation_test.to_csv('data/2015/process/Val_Test_Set.csv', index=False)
# validation.to_csv('data/2015/process/Val_Set.csv', index=False)
# test.to_csv('data/2015/process/Test_Set.csv', index=False)
#
# print("CSV file has been saved.")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import qmc

#设置显示窗口数据显示完整
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def latin_hypercube_split(data, test_size=0.15, random_state=None):
    """
    Perform Latin Hypercube Sampling (LHS) to split a 2D array into training and testing sets.

    Parameters:
    - data: ndarray
        A 2D numpy array where each row represents a sample.
    - test_size: float
        Proportion of the data to include in the test split (default is 0.15).
    - random_state: int, optional
        Random seed for reproducibility.

    Returns:
    - train_set: ndarray
        The training set (1 - test_size of the rows).
    - test_set: ndarray
        The test set (test_size of the rows).
    """
    # Ensure reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Get number of samples and dimensions
    n_samples, n_features = data.shape

    # Initialize Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=n_features)

    # Generate Latin Hypercube samples
    lhs_samples = sampler.random(n=n_samples)

    # Map LHS samples to row indices
    row_indices = (lhs_samples[:, 0] * n_samples).astype(int)  # Use the first dimension to select rows

    # Determine test set size
    test_set_size = int(test_size * n_samples)

    # Get test and train indices
    test_indices = row_indices[:test_set_size]
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

    return train_indices, test_indices


# Load the dataset
data_path = "data/2015/process/Spe_OC_N_CaCO3_pH.csv"  # Replace with your file path
data = pd.read_csv(data_path)

# Separate spectral data and soil properties
spectral_data = data.iloc[:, 1:-4]  # Spectral absorbance values
soil_properties = data.iloc[:, -4:]  # OC, N, CaCO3, pH

# Normalize spectral data and soil properties
scaler = MinMaxScaler()
spectral_data_normalized = scaler.fit_transform(spectral_data)
soil_properties_normalized = scaler.fit_transform(soil_properties)

# Combine normalized spectral data and soil properties
combined_data = np.hstack((spectral_data_normalized, soil_properties_normalized))

train_indices, test_indices = latin_hypercube_split(combined_data, test_size=0.15, random_state=42)

# Convert numpy arrays to DataFrames and assign column names
train_set_df = pd.DataFrame(data.to_numpy()[train_indices, 1:], columns=np.concatenate([spectral_data.columns, soil_properties.columns]))
test_set_df = pd.DataFrame(data.to_numpy()[test_indices, 1:], columns=np.concatenate([spectral_data.columns, soil_properties.columns]))

# Function to calculate statistics for each soil property
def calculate_statistics(df, properties):
    stats = {}
    for col in properties:
        series = df[col]
        stats[col] = {
            "Count": series.count(),
            "Min": series.min(),
            "Max": series.max(),
            "Q25": series.quantile(0.25),
            "Median": series.median(),
            "Q75": series.quantile(0.75),
            "Mean": series.mean(),
            "Std": series.std(),
            "Skewness": skew(series),
            "CV (%)": (series.std() / series.mean()) * 100 if series.mean() != 0 else 0,
        }
    return pd.DataFrame(stats).T

# Calculate statistics for the entire dataset, training set, and test set
statistics_all = calculate_statistics(soil_properties, soil_properties.columns)
statistics_train = calculate_statistics(train_set_df.iloc[:, -4:], soil_properties.columns)  # Last 4 columns
statistics_test = calculate_statistics(test_set_df.iloc[:, -4:], soil_properties.columns)  # Last 4 columns

# Print statistics for review
print("Statistics for All Data:\n", statistics_all)
print("Statistics for Training Set:\n", statistics_train)
print("Statistics for Test Set:\n", statistics_test)

# List of soil property names to use in titles and x labels
soil_properties_list = soil_properties.columns



import seaborn as sns
import matplotlib.pyplot as plt

# 设置绘图风格为白色背景，这将移除网格
# sns.set(style="white")

# 假设 df, train_set, test_set 已经定义并包含相应的数据
soil_attributes = ['OC', 'N', 'CaCO3', 'pH(H2O)']
soil_attributes_title = ['OC', 'N', r'CaCO$_3$', r'pH(H$_2$O)']

# 设置调色板，让图形看起来更美观
# palette = sns.color_palette("husl", 3)  # 使用更柔和的颜色方案
colors = {
    "Total Dataset": "skyblue",
    "Training Set": "lightgreen",
    "Test Set": "salmon"
}

# 绘制每个属性的 KDE 图
for i, attribute in enumerate(soil_attributes):
    plt.figure(figsize=(8, 6))

    sns.kdeplot(soil_properties[attribute], label="Total Dataset", color=colors["Total Dataset"], fill=True, alpha=0.4)
    sns.kdeplot(train_set_df[attribute], label="Training Set", color=colors["Training Set"], fill=True, alpha=0.6)
    sns.kdeplot(test_set_df[attribute], label="Test Set", color=colors["Test Set"], fill=True, alpha=0.8)

    plt.xlabel(soil_attributes_title[i] + f'\n({chr(97 + i)})', fontsize=20)
    plt.ylabel('Density', fontsize=20)

    plt.legend(fontsize=16, loc='upper right')
    plt.grid(False)

    plt.tick_params(axis="both", labelsize=18)

    # 调整布局，避免标签重叠
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'data/2015/picture/{attribute} KDE.png', dpi=600)

# Plot histograms for comparison with improved aesthetics and labels
# for idx, col in enumerate(soil_properties_list):
#     plt.figure(figsize=(10, 8))
#
#     # Plot histograms with customized colors and transparency
#     plt.hist(soil_properties[col], bins=30, alpha=0.6, label='All Data', color='royalblue', edgecolor='black',
#              linewidth=1.2)
#     plt.hist(train_set_df[col], bins=30, alpha=0.6, label='Training Set', color='forestgreen', edgecolor='black',
#              linewidth=1.2)
#     plt.hist(test_set_df[col], bins=30, alpha=0.6, label='Test Set', color='darkorange', edgecolor='black',
#              linewidth=1.2)
#
#     # Add title, labels, and grid
#     # plt.title(f"Distribution of {col} ({chr(97 + idx)})", fontsize=16, fontweight='bold')
#     plt.xlabel(f'({chr(97 + idx)}) {col}', fontsize=22)  # Adding (a), (b), (c), (d)
#     plt.ylabel('Frequency', fontsize=22)
#     plt.legend(loc='upper right', fontsize=18)
#
#     plt.tick_params(axis='both', labelsize=18)  # 这里设置字体大小为18，可以根据需要调整
#     # Add grid
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # Adjust layout and show the plot
#     plt.tight_layout()
#     plt.show()
    # plt.savefig(f'data/2015/picture/{col} histogram.png', dpi=600)


from sklearn.decomposition import PCA

# Apply PCA to reduce spectral data to 2 dimensions
pca = PCA(n_components=2)   # Reduce to 2 components
spectral_data_pca = pca.fit_transform(spectral_data)  # Apply PCA to the entire spectral data
train_set_pca = pca.transform(train_set_df.iloc[:, :-4])  # Apply PCA to training set
test_set_pca = pca.transform(test_set_df.iloc[:, :-4])  # Apply PCA to test set

# Convert PCA results to DataFrame for easier handling
spectral_data_pca_df = pd.DataFrame(spectral_data_pca, columns=['PCA1', 'PCA2'])
train_set_pca_df = pd.DataFrame(train_set_pca, columns=['PCA1', 'PCA2'])
test_set_pca_df = pd.DataFrame(test_set_pca, columns=['PCA1', 'PCA2'])

# Print explained variance ratio for the two principal components
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance ratio of PCA1: {explained_variance_ratio[0]:.2%}")
print(f"Explained variance ratio of PCA2: {explained_variance_ratio[1]:.2%}")

# Plot histograms for PCA1 and PCA2
for idx, col in enumerate(['PCA1', 'PCA2']):
    plt.figure(figsize=(8, 6))

    # Plot histograms with customized colors and transparency
    # plt.hist(spectral_data_pca_df[col], bins=30, alpha=0.6, label='All Data', color='royalblue', edgecolor='black',
    #          linewidth=1.2)
    # plt.hist(train_set_pca_df[col], bins=30, alpha=0.6, label='Training Set', color='forestgreen', edgecolor='black',
    #          linewidth=1.2)
    # plt.hist(test_set_pca_df[col], bins=30, alpha=0.6, label='Test Set', color='darkorange', edgecolor='black',
    #          linewidth=1.2)

    sns.kdeplot(spectral_data_pca_df[col], label="Total Dataset", color=colors["Total Dataset"], fill=True, alpha=0.4)
    sns.kdeplot(train_set_pca_df[col], label="Training Set", color=colors["Training Set"], fill=True, alpha=0.6)
    sns.kdeplot(test_set_pca_df[col], label="Test Set", color=colors["Test Set"], fill=True, alpha=0.8)

    # Add title, labels, and grid
    plt.xlabel(f'Spectral {col} Value\n({chr(97 + idx)})', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(loc='upper right', fontsize=16)

    plt.tick_params(axis='both', labelsize=18)  # 这里设置字体大小为18，可以根据需要调整

    plt.grid(False)

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'data/2015/picture/spectral {col} KED.png', dpi=600)
