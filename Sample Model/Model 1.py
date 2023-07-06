import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# I convereted the .dat file to a .csv file for ease of use
data = pd.read_csv('pscalar_0p0018.csv')
time = data.iloc[:, 0]
data_values = data.iloc[:, 1]

# Calculating the mean value for each unique time
unique_times = np.unique(time)
data_mean = []
for t in unique_times:
    indices = np.where(time == t)[0]
    if len(indices) > 0:
        mean_value = np.sum(data_values[indices]) / len(indices)
        data_mean.append(mean_value)

x = unique_times
y = np.array(data_mean)

# Determing the centre point for the polynomial fit
center = (x.max() - x.min()) / 2

# Shifting the x values to be centered at new center
x_shifted = x - center

# Model function with the shifted x values
def model_func(x, m, a, b):
    return (m**2) * np.exp(-m * x) + (m**2) * np.exp(m * x)

# fit
params, _ = curve_fit(model_func, x_shifted, y)

# Model predictions using the shifted x values
model_predictions = model_func(x_shifted, *params)

# Calculating the residuals
residuals = y - model_predictions
ss_residuals = np.sum(residuals**2)
ss_total = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_residuals / ss_total)

# Calculating the chi-square value
chi_squared = np.sum(residuals**2 / y)

plt.figure(figsize=(10, 8))
# Data and Model Plot
plt.subplot(2, 1, 1)
plt.scatter(x, y, label='Data')
plt.plot(x, model_predictions, color='red', label='Model')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.grid(True)

# Residual Plot
plt.subplot(2, 1, 2)
plt.scatter(x, residuals, color='blue', label='Residuals')
plt.axhline(0, color='red')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)

# Chi-square value and R-squared value
plt.figtext(0.20, 0.80, f'Chi-Square: {chi_squared:.2e}', fontsize=12)
plt.figtext(0.20, 0.75, f'R-squared: {r_squared:.4f}', fontsize=12)
plt.tight_layout()
plt.show()


#print('R-squared:', r_squared)
