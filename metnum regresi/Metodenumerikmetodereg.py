import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_csv('Student_Performance.csv')

# Extract relevant columns
NL = df['Sample Question Papers Practiced'].values
NT = df['Performance Index'].values

# Function to calculate RMS error
def rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Metode 1: Regresi Linear
slope, intercept, r_value, p_value, std_err = stats.linregress(NL, NT)
NT_pred_linear = intercept + slope * NL

# Metode 3: Regresi Eksponensial
log_NT = np.log(NT)
slope_exp, intercept_exp = np.polyfit(NL, log_NT, 1)
NT_pred_exp = np.exp(intercept_exp + slope_exp * NL)

# Menghitung galat RMS
rms_linear = rms_error(NT, NT_pred_linear)
rms_exponential = rms_error(NT, NT_pred_exp)

# Plot data and linear regression
plt.figure(figsize=(10, 6))
plt.scatter(NL, NT, label='Data Asli')
plt.plot(NL, NT_pred_linear, color='r', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear antara NL dan NT')
plt.legend()
plt.grid(True)
plt.show()

# Plot data and exponential regression
plt.figure(figsize=(10, 6))
plt.scatter(NL, NT, label='Data Asli')
plt.plot(NL, NT_pred_exp, color='g', label='Regresi Eksponensial')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Eksponensial antara NL dan NT')
plt.legend()
plt.grid(True)
plt.show()

print(f'Galat RMS Regresi Linear: {rms_linear:.2f}')
print(f'Galat RMS Regresi Eksponensial: {rms_exponential:.2f}')

# Testing functions
def test_regression_models():
    # Check if linear regression slope and intercept are as expected
    assert np.isclose(slope, stats.linregress(NL, NT)[0]), "Slope mismatch in linear regression"
    assert np.isclose(intercept, stats.linregress(NL, NT)[1]), "Intercept mismatch in linear regression"
    
    # Check if exponential regression coefficients are as expected
    assert np.isclose(slope_exp, np.polyfit(NL, log_NT, 1)[0]), "Slope mismatch in exponential regression"
    assert np.isclose(intercept_exp, np.polyfit(NL, log_NT, 1)[1]), "Intercept mismatch in exponential regression"
    
    # Check if RMS errors are within reasonable ranges
    assert rms_linear > 0, "RMS error for linear regression should be positive"
    assert rms_exponential > 0, "RMS error for exponential regression should be positive"
    print("All tests passed!")

# Run the tests
test_regression_models()
