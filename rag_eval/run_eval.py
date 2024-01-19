import logging

import giskard
import pandas as pd

from model.rag_model import RagModel


def generate_html_report(llm_model, report_path):
    """Wrapper function to generate a HTML Giskard report."""
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
    full_report = giskard.scan(giskard_model, giskard_dataset)
    full_report.to_html(report_path)


def run_eval(local=False, llm_model="dolly", report_path="scan_report.html"):
    """Entry point to run evaluation of the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation pipeline")
    if local:
        logger.info("Generating local report...")
        generate_html_report(llm_model=llm_model, report_path=report_path)
    else:
        print("Nothing to see here!")
