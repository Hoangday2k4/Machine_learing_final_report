import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrixPlot:
    def __init__(self, figsize=(6, 5)):
        """
        Khởi tạo đối tượng ConfusionMatrixPlot.

        Parameters:
        - figsize (tuple): Kích thước của biểu đồ (width, height).
        """
        self.figsize = figsize

    def plot(self, conf_matrix, labels, title='Ma trận nhầm lẫn', cmap='Blues'):
        """
        Vẽ ma trận nhầm lẫn dưới dạng heatmap.

        Parameters:
        - conf_matrix (np.ndarray): Ma trận nhầm lẫn (confusion matrix).
        - labels (list): Danh sách nhãn cho trục x và y.
        - title (str): Tiêu đề biểu đồ.
        - cmap (str): Bảng màu heatmap (mặc định: 'Blues').
        """
        plt.figure(figsize=self.figsize)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title(title)
        plt.show()