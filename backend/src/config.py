"""Central configuration for the Windarling Fe-grade soft sensor (backend)."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = BACKEND_DIR / "data" / "raw" / "Windarling_Ore_Composition.csv"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "fe_soft_sensor.joblib"

TARGET = "Fe"
CAT_COL = "Lithotype"
CHEM_COLS = ["P", "SiO2", "Al2O3", "S", "Mn", "CL", "LOI"]
# Components treated as "non-iron-oxide" mass in the mass-balance proxy.
GANGUE_COLS = ["SiO2", "Al2O3", "P", "S", "Mn", "CL"]
COORD_COLS = ["Easting", "Northing"]
ID_COL = "Hole_id"
INDICATOR_COLS = ["Sample.West", "Sample.East", "West", "East"]

REQUIRED_RAW_COLS = [ID_COL, *INDICATOR_COLS, *COORD_COLS, CAT_COL, TARGET, *CHEM_COLS]
# Minimum columns a user must supply for prediction (Fe is optional - measured grade).
REQUIRED_INPUT_COLS = [*CHEM_COLS, CAT_COL]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
INTERVAL_LEVEL = 0.95

DATASET_CITATION = (
    "Windarling Ore Composition dataset - R package gmGeostats; original data from "
    "Ward (2015), MSc thesis, Edith Cowan University. Licence: CC BY-SA 4.0 (as stated "
    "in the project brief)."
)
