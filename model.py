import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("bridge_digital_twin_dataset(small).csv")

# Target column
target = "Structural_Health_Index_SHI"

# Features
features = [
    "Strain_microstrain",
    "Deflection_mm",
    "Vibration_ms2",
    "Tilt_deg",
    "Temperature_C",
    "Humidity_percent",
    "Wind_Speed_ms",
    "Vehicle_Load_tons"
]
# Clean dataset
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
