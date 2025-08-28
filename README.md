# Thermal Conductivity ANN (Offshore Wind)

ANN (MLPRegressor) predicting **thermal conductivity** from degree of saturation (Sr), porosity (n), and Class, based on an Imperial College London UROP project.

## Quickstart

```bash
pip install -r requirements.txt

# Train (uses your Excel file)
python -m thermal_ann.train --excel "data/ASTM_classification.xlsx"
# Artifacts → models/, plots & metrics → results/

# Predict on new CSV with columns Sr,n,Class
python -m thermal_ann.predict --csv data/new_samples.csv --out predictions.csv
