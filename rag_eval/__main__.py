import argparse

from .run_eval import run_eval

parser = argparse.ArgumentParser()
parser.add_argument("--local", help="Generate local HTML report", action="store_true")
parser.add_argument(
    "--llm",
    help="Choose LLM to use in the RAG model. Supported models : 'openai' or 'dolly' ",
    default="dolly",
)
parser.add_argument(
    "--report_path", help="Path for local HTML report ", default="scan_report.html"
)

args = parser.parse_args()

run_eval(local=args.local, llm_model=args.llm, report_path=args.report_path)
