import logging

import dotenv
import giskard
import pandas as pd
from dotenv import dotenv_values
from giskard import GiskardClient, Suite, demo, testing

from model.classification_model import ClassificationModel
from model.rag_model import RagModel


def get_giskard_client() -> GiskardClient:
    """Generate a Giskard Client to upload tests to Hub."""
    url = "http://localhost:19000"
    config = dotenv_values(".env")
    api_key = config["GISKARDHUB_API_KEY"]

    # Create a giskard client to communicate with Hub
    client = GiskardClient(url=url, key=api_key)

    return client


def generate_rag_giskard_objects(llm_model) -> tuple:
    """Generate RAG objects."""
    model = RagModel(document_path="./data/us_sou_2023.txt", llm_model=llm_model)
    model.initialize_rag_chain()
    questions_df = pd.read_csv("./data/questions.csv")

    # Generate a Giskard Model
    giskard_model = giskard.Model(
        model=model.predict,
        model_type="text_generation",
        name="State of the Union 2023 Question Answering",
        description="This model answers any question about the 2023 US State of the Union Speech by Joe Biden.",
        feature_names=["question"],
    )

    giskard_dataset = giskard.Dataset(questions_df, target=None)

    return giskard_model, giskard_dataset


def generate_titanic_giskard_objects() -> tuple:
    """Generate objects associated with Titanic model."""
    model = ClassificationModel()
    _ = model.model_init()

    data = demo.titanic_df()

    # Generate a Giskard Model
    giskard_model_titanic = giskard.Model(
        model=model.predict,
        model_type="classification",  # Either regression, classification or text_generation.
        name="Titanic model",  # Optional
        classification_labels=model.classes_,
        # Their order MUST be identical to the prediction_function's output order
        feature_names=[
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
        # Default: all columns of your dataset
        # classification_threshold=0.5,  # Default: 0.5
    )
    giskard_dataset_titanic = giskard.Dataset(
        df=data,
        target="Survived",  # Ground truth variable
        name="Titanic dataset",  # Optional
        cat_columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
        # List of categorical columns. Optional, but is a MUST if available. Inferred automatically if not.
    )

    return giskard_model_titanic, giskard_dataset_titanic


def generate_html_report(model, llm_model, report_path):
    """Wrapper function to generate a HTML Giskard report."""
    if model == "rag":
        giskard_model, giskard_dataset = generate_rag_giskard_objects(llm_model)
    elif model == "titanic":
        giskard_model, giskard_dataset = generate_titanic_giskard_objects()
    else:
        raise ValueError(
            'Invalid model name passed. Run "python -m rag_eval --help" for more details on list of handled models.'
        )

    full_report = giskard.scan(giskard_model, giskard_dataset)
    full_report.to_html(report_path)


def upload_to_hub(project_key):
    """Wrapper function to upload objects to Hub."""
    giskard_model_titanic, giskard_dataset_titanic = generate_titanic_giskard_objects()

    giskard_client = get_giskard_client()

    giskard_dataset_titanic.upload(giskard_client, project_key)

    test_suite = (
        Suite(name="Titanic test suite")
        .add_test(testing.test_f1(model=giskard_model_titanic))
        .add_test(testing.test_accuracy(model=giskard_model_titanic))
    )
    test_suite.upload(giskard_client, project_key)

    giskard_model_titanic.upload(giskard_client, project_key)


def run_eval(
    model,
    to_hub=False,
    llm_model="openai",
    report_path="scan_report.html",
    project_key="rag_eval",
):
    """Entry point to run evaluation of the model."""
    dotenv.load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation pipeline for model '{model}'")
    if not to_hub:
        logger.info("Generating local report...")
        generate_html_report(model=model, llm_model=llm_model, report_path=report_path)
    else:
        if model == "titanic":
            upload_to_hub(project_key=project_key)
        else:
            raise ValueError(
                'RAG model currently not handled with Hub. Run "python -m rag_eval --help" for more details.'
            )
