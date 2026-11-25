"""
run_pipeline.py  —  one command to build the whole RentWise UK system (4 modules).

Every phase and every module by composing the classes I built.
Running this reproduces all data, models, charts and reports from scratch. It is
also the clearest demonstration of the object-oriented design: each stage is a
class with a run() method, and here I simply run them in order.
"""

# Module 2 pipeline (rental price & affordability) - the fully-worked example.
import config
from src.data.make_sample_data import SampleDataGenerator
from src.data.real_data import RealDataLoader
from src.data.clean_data import DataCleaner
from src.features.build_features import FeatureBuilder
from src.analysis.eda import DataExplorer
from src.models.train_baseline import BaselineTrainer
from src.models.train_advanced import AdvancedTrainer
from src.models.evaluate import ModelEvaluator
from src.models.explain_shap import ShapExplainer
from src.models.affordability import AffordabilityEngine
# Sibling modules 1, 3, 4.
from src.modules.module1_sale_price import SalePriceModule
from src.modules.module3_geospatial import GeospatialModule
from src.modules.module4_forecasting import ForecastingModule


class Pipeline:
    """run every phase of Module 2, then the three sibling modules, in order."""

    # pick the Phase-1 data source from config.DATA_MODE. Everything after it is
    # identical, because both sources write the same rentwise_sample.csv schema.
    _SOURCE = RealDataLoader if config.DATA_MODE == "real" else SampleDataGenerator
    _SOURCE_LABEL = ("Module 2  Phase 1  Load REAL data" if config.DATA_MODE == "real"
                     else "Module 2  Phase 1  Generate sample data")

    STAGES = [
        (_SOURCE_LABEL, _SOURCE),
        ("Module 2  Phase 2  Clean data", lambda: DataCleaner(mode="synthetic")),
        ("Module 2  Phase 4  Build features", FeatureBuilder),
        ("Module 2  Phase 3  Explore (EDA)", DataExplorer),
        ("Module 2  Phase 5a Baseline model", BaselineTrainer),
        ("Module 2  Phase 5b Advanced models", AdvancedTrainer),
        ("Module 2  Phase 6  Evaluate", ModelEvaluator),
        ("Module 2  Phase 7  SHAP explainability", ShapExplainer),
        ("Module 2  Phase 8  Affordability engine", AffordabilityEngine),
        ("Module 1  Sale Price Prediction", SalePriceModule),
        ("Module 3  Geospatial Driver Analysis", GeospatialModule),
        ("Module 4  Regional Trend Forecasting", ForecastingModule),
    ]

    def run(self):
        print(f"[Pipeline] DATA_MODE = {config.DATA_MODE!r} "
              f"({'real ONS/HMLR files' if config.DATA_MODE == 'real' else 'synthetic development data'})")
        for title, StageClass in self.STAGES:
            print("\n" + "=" * 60)
            print(title)
            print("=" * 60)
            StageClass().run()
        print("\n" + "=" * 60)
        print("All four modules built. Launch the dashboard with:")
        print("   streamlit run app/streamlit_app.py")
        print("=" * 60)


if __name__ == "__main__":
    Pipeline().run()
