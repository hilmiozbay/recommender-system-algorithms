import numpy as np


# Sample data
x = np.array([10, 20, 30, 40, 50])
y = np.array([5, 15, 25, 35, 45])

# Pearson correlation using NumPy
corr = np.corrcoef(x, y)[0, 1]
print("Pearson Correlation (NumPy):", corr)


# Manual calculation of Pearson correlation
def pearson_correlation(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_sq = np.sum(x**2)
    sum_y_sq = np.sum(y**2)
    sum_xy = np.sum(x * y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))

    if denominator == 0:
        return 0
    return numerator / denominator

manual_corr = pearson_correlation(x, y)
print("Pearson Correlation (Manual):", manual_corr)
