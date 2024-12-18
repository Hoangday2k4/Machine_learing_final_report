import numpy as np
import matplotlib.pyplot as plt

class ScatterPlot:
    def __init__(self, figsize=(8, 6)):
        """
        Khởi tạo đối tượng ScatterPlot.

        Parameters:
        - figsize (tuple): Kích thước của biểu đồ (width, height).
        """
        self.figsize = figsize

    def plot_1(self, x, y, c='blue', edgecolor='k', alpha=0.7, title='', xlabel='', ylabel='', grid=True):
        """
        Vẽ biểu đồ scatter plot.

        Parameters:
        - x (np.ndarray or list): Tọa độ x của các điểm.
        - y (np.ndarray or list): Tọa độ y của các điểm.
        - c (str or list): Màu sắc của các điểm (mặc định: 'blue').
        - edgecolor (str): Màu sắc viền của các điểm (mặc định: 'k').
        - alpha (float): Độ trong suốt của các điểm (mặc định: 0.7).
        - title (str): Tiêu đề biểu đồ.
        - xlabel (str): Nhãn trục x.
        - ylabel (str): Nhãn trục y.
        - grid (bool): Hiển thị lưới (mặc định: True).
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(x, y, c=c, edgecolor=edgecolor, alpha=alpha)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
        plt.show()

    def plot_2(self, x, y, c,cmap = 'viridis', edgecolor='k', alpha=0.7, title='', xlabel='', ylabel='', grid=True):
        """
        Vẽ biểu đồ scatter plot.

        Parameters:
        - x (np.ndarray or list): Tọa độ x của các điểm.
        - y (np.ndarray or list): Tọa độ y của các điểm.
        - c (str or list): Màu sắc của các điểm (mặc định: 'blue').
        - edgecolor (str): Màu sắc viền của các điểm (mặc định: 'k').
        - alpha (float): Độ trong suốt của các điểm (mặc định: 0.7).
        - title (str): Tiêu đề biểu đồ.
        - xlabel (str): Nhãn trục x.
        - ylabel (str): Nhãn trục y.
        - grid (bool): Hiển thị lưới (mặc định: True).
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(x, y, c=c,cmap = cmap, edgecolor=edgecolor, alpha=alpha)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label='Nhãn phân cụm')
        plt.grid(grid)
        plt.show()