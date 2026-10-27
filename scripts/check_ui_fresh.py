"""Exercise real form submissions and capture only their new CLI responses."""

import argparse
import json
from pathlib import Path

from ml_training.common import ROOT, write_json
from ml_training.ui_evidence import validate_submission


def main():
    from streamlit.testing.v1 import AppTest

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demos",
        nargs="+",
        choices=["distillation", "schema_dsl", "rlvr", "edge", "retrieval", "explainability", "hybrid"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = []
    for demo in args.demos:
        app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=240).run()
        app.sidebar.selectbox[0].select(demo).run()
        assert not app.exception, (demo, app.exception)

        def value(elements, label):
            return next(item.value for item in elements if item.label == label)

        run_name = value(app.selectbox, "Run")
        if demo == "hybrid":
            button, variant = "Run pipeline", value(app.selectbox, "Pipeline")
            submitted = {
                "id": "interactive",
                "text": value(app.text_area, "Request"),
                "version": value(app.selectbox, "API version"),
                "resource": value(app.selectbox, "Resource"),
                "operation": value(app.selectbox, "API operation"),
                "role": value(app.selectbox, "Permission"),
                "query_backend": value(app.selectbox, "Query model"),
                "transaction_row": int(value(app.number_input, "Held-out transaction row")),
                "metadata": json.loads(value(app.text_area, "Request metadata vector")),
            }
        elif demo == "explainability":
            button, variant = "Recompute prediction and explanation", None
            submitted = {
                "row_index": int(value(app.number_input, "Held-out row")),
                "amount": float(value(app.number_input, "Changed amount")),
            }
        else:
            selector = next(item for item in app.selectbox if item.label == "Variant")
            variant = (
                "tuned_student"
                if demo == "distillation"
                else "tuned"
                if demo in ["schema_dsl", "edge"]
                else "rlvr"
                if demo == "rlvr"
                else "tuned_bi"
            )
            selector.select(variant)
            button, submitted = "Run selected variant", value(app.text_area, "Request")
        # A stale session value or the evidence tab cannot satisfy this check.
        app.session_state["fresh_execution"] = None
        next(item for item in app.button if item.label == button).click().run()
        assert not app.exception, (demo, app.exception)
        assert not app.error, (demo, [item.value for item in app.error])
        execution = app.session_state["fresh_execution"]
        result = validate_submission(execution, demo, submitted, variant, run_name)
        observed.append(
            {
                "demo": demo,
                "button": button,
                "execution_source": "submitted_cli",
                "command": execution["command"],
                "submitted": submitted,
                "variant": variant,
                "run_name": run_name,
                "result": result,
            }
        )
        write_json(
            args.output,
            {
                "status": "fresh_form_submissions_passed",
                "completed": observed,
                "scope": "AppTest captured the new CLI result from this exact form submission and checked its request, variant and run. Saved evidence-tab examples cannot count as fresh execution.",
            },
        )
        print(f"Fresh UI submission passed: {demo}", flush=True)


if __name__ == "__main__":
    main()
