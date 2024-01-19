import argparse

from .run_eval import run_eval

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="Chose model for which to run evaluation. Two models supported : 'rag' or 'titanic",
)
parser.add_argument(
    "--to_hub",
    help="Generate tests on Hub. Only supported for titanic model",
    action="store_true",
)
parser.add_argument(
    "--llm",
    help="Choose LLM to use in the RAG model. Supported models : 'openai'",
    default="openai",
)
parser.add_argument(
    "--report_path", help="Path for local HTML report ", default="scan_report.html"
)
parser.add_argument(
    "--project_key",
    help="Project key to upload objects to on the Hub",
    default="rag_eval",
)

args = parser.parse_args()

run_eval(
    model=args.model,
    to_hub=args.to_hub,
    llm_model=args.llm,
    report_path=args.report_path,
    project_key=args.project_key,
)
