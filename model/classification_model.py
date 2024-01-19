import pandas as pd
from giskard import demo


class ClassificationModel:
    """Classification Model."""

    def __init__(self):
        self.model = None
        self.classes_ = None

    @classmethod
    def preprocessing_function(cls):
        pass

    def model_init(self):
        """Init Titanic model from demo."""
        _, demo_sklearn_model = demo.titanic_pipeline()
        self.model = demo_sklearn_model
        self.classes_ = demo_sklearn_model.classes_

    def predict(self, df: pd.DataFrame):
        """Predict function of the model."""
        if not self.model:
            self.model_init()
        preprocessing_function, _ = demo.titanic_pipeline()
        preprocessed_df = preprocessing_function(df)
        return self.model.predict_proba(preprocessed_df)
