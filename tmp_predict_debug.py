import traceback
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
try:
    data = CustomData(
        "female",
        "group A",
        "bachelor's degree",
        "standard",
        "none",
        80.0,
        75.0,
    )
    df = data.get_data_as_DataFrame()
    print(df)
    pipe = PredictPipeline()
    print(pipe.predict(df))
except Exception:
    traceback.print_exc()
