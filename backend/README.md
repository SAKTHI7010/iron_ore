# Windarling Iron-Ore Fe-Grade Soft Sensor

Two folders only:

```
backend/   data, ML code (src/), trained model, reports, tests
frontend/  Streamlit dashboard (app.py)
```

## Setup
```bash
pip install -r backend/requirements.txt
```

## Train / evaluate (from `backend/`)
```bash
cd backend
python -m src.eda            # EDA summary + figures -> reports/
python -m src.train_model    # tune on train split, evaluate on hold-out, save models/fe_soft_sensor.joblib
python -m pytest -q          # 18 tests (data, preprocessing, prediction, Streamlit smoke tests)
```

## Run the dashboard (from `frontend/`)
```bash
cd frontend
streamlit run app.py
```
If the model file is missing the app trains it automatically on first start.

## Method (Project 1 brief)
* Data: `backend/data/raw/Windarling_Ore_Composition.csv` (1,600 x 16, mass fractions -> shown as %; raw file untouched).
* Features: P, SiO2, Al2O3, S, Mn, CL, LOI + one-hot Lithotype. Coordinates / ids are **excluded** from the model
  (used only for the Digital Twin map).
* Preprocessing (fit on train only): median imputation by lithotype, optional 1st/99th-percentile capping
  (chosen by training-CV; the CV rejected capping), one-hot encoding, scaling where needed (Ridge).
* Models: mechanistic baseline `Fe = a + b*(100 - gangue) + c*LOI`, Ridge, XGBoost, and the hybrid
  (baseline + XGBoost on out-of-fold residuals). Selection/tuning uses 5-fold CV on the training split only.
* Uncertainty: 95 % interval from empirical quantiles of cross-validated hybrid errors.
* Explainability: additive attribution = linear-baseline contribution + TreeSHAP of the correction.
* Digital Twin: static spatial Fe map (Easting/Northing) that updates with pinned/batch samples.

## Notes and limitations
* The brief lists lithotypes basalt/goethite/magnetite/schist; the real data contain **goethite, hematite, schist, basalt**
  (no magnetite). Basalt has 1 sample and schist 17, so per-class results for them are unreliable.
* Fe std in the data is ~5.8 %, not the 3.5-3.7 % quoted in the brief (verified, not hard-coded).
* R² is ~0.998 for every model because the chemistry is compositional (closure); compare RMSE/MAE.
* The interval has constant width and under-covers schist.

## Deployment
Deploy `frontend/app.py` on Streamlit Community Cloud (or any container) with the repository root as the
working tree, `backend/` alongside `frontend/`, and `pip install -r backend/requirements.txt`.
Dataset licence: CC BY-SA 4.0 (Ward 2015, gmGeostats) - keep attribution.
