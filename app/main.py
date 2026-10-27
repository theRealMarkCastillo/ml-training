"""Local comparison UI over the same immutable artifacts used by the CLI."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

import pandas as pd
import streamlit as st

from ml_training.common import ROOT, environment_executable, environments, read_json, verify_run
from ml_training.processes import run_isolated

TRACKS = {
    "distillation": "D01 · Distilled support routing",
    "schema_dsl": "D02 · Structured queries",
    "rlvr": "D03 · Executable-reward training",
    "edge": "D04 · Offline device interpreter",
    "tabular": "D05 · Tabular prediction",
    "fast_classifier": "D06 · Fast request classification",
    "explainability": "D07 · Prediction explanations",
    "retrieval": "D08 · Retrieval and reranking",
    "hybrid": "D09 · Hybrid assistant",
}
QUERY_DEFAULTS = {
    "distillation": "The card was charged twice. Account C1234.",
    "schema_dsl": "Count tickets where priority equals urgent AND amount is greater than 50.",
    "rlvr": "Count tickets where age_hours is greater than 24.",
    "edge": "Set the thermostat to 22 degrees.",
    "retrieval": "Inspect the v2 thermostat settings without changing them.",
}


def inventory():
    runs = []
    for path in sorted((ROOT / "artifacts").glob("*/*/run.json"), reverse=True):
        try:
            info = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue  # A running experiment may be writing its manifest; refresh later.
        if info.get("demo") in TRACKS:
            runs.append({"path": path.parent, **info})
    return runs


def scalar_table(metrics):
    if isinstance(metrics, dict) and all(isinstance(value, dict) for value in metrics.values()):
        rows = [{"variant": name, **{key: value for key, value in values.items()
                                     if value is None or isinstance(value, (str, int, float, bool))}}
                for name, values in metrics.items()]
        return pd.DataFrame(rows)
    return None


def execute_command(command):
    st.session_state["fresh_execution"] = None
    with st.spinner("Loading saved models and executing this request…"):
        try:
            result = run_isolated(command,cwd=ROOT,timeout=180,
                env=os.environ | {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
            if result.returncode:
                st.error(result.stderr[-6000:] or result.stdout[-6000:])
                return None
            value = json.loads(result.stdout)
            st.session_state["fresh_execution"] = {"command":command,"result":value}
            return value
        except (subprocess.TimeoutExpired,OSError,ValueError) as error:
            st.error(f"Fresh execution did not complete: {error}")
            return None


st.set_page_config(page_title="ML Training Demo Lab", layout="wide")
st.title("ML Training Demo Lab")
st.caption("Observed experiments · synthetic data · local models · simulated device actions")
all_runs = inventory()
completed = {r["demo"] for r in all_runs if r["status"] == "evaluated"}
left, middle, right = st.columns(3)
left.metric("Evaluated tracks", f"{len(completed)} / 9")
middle.metric("Source use cases", "8")
right.metric("Production deployments", "0")
st.write("Compare saved evidence, inspect failures, and run a fresh model prediction. A completed experiment may show that training did not help. Recorded results and fresh execution are labeled separately.")

with st.sidebar:
    st.header("Explore the lab")
    demo = st.selectbox("Demo", list(TRACKS), format_func=TRACKS.get)
    include_smoke = st.checkbox("Include smoke and failed attempts", value=False)
    if st.button("Refresh run inventory"):
        st.rerun()
    st.caption("Artifacts stay on this computer. The interface reads the same run directories as the CLI.")

options = [r for r in all_runs if r["demo"] == demo and (include_smoke or r["status"] == "evaluated")]
if not options:
    st.info(f"{TRACKS[demo]} has no completed experiment available yet. Its implementation/evaluation remains incomplete.")
    st.stop()
selected_name = st.selectbox("Run", [r["path"].name for r in options])
selected = next(r for r in options if r["path"].name == selected_name)
run = selected["path"]
st.subheader(TRACKS[demo])
st.caption(f"Status: {selected['status']} · Profile: {selected['profile']} · Seed: {selected['seed']} · {run.relative_to(ROOT)}")
if selected["status"] not in ["evaluated", "smoke"]:
    st.error(selected.get("evidence_rejected_reason", selected.get("error", "This run has not completed.")))
    st.json(selected)
    st.stop()
if selected["status"] == "smoke":
    st.info("This is wiring and compatibility evidence from a tiny run. It does not replace the full experiment.")

results_tab, try_tab, evidence_tab = st.tabs(["Recorded results", "Fresh prediction", "Evidence and failures"])
with results_tab:
    metrics = read_json(run / "metrics.json") if (run / "metrics.json").exists() else {}
    st.markdown("**Held-out evaluation**")
    table = scalar_table(metrics)
    if demo == "explainability":
        st.write(f"Explained {selected['rows_explained']} held-out rows. Maximum additivity error: {selected['max_additivity_residual']:.3g}; declared tolerance: {selected['tolerance']:.1g}.")
        st.caption("Mean absolute TreeSHAP contribution over these rows, in raw log-odds. The background contains 100 training rows; attribution uses interventional perturbation.")
        contributions = read_json(run / "global_attribution.json")
        st.dataframe(pd.DataFrame([{"feature":key,"mean_absolute_contribution":value} for key,value in
                                  sorted(contributions.items(),key=lambda pair:-pair[1])]),hide_index=True)
    elif demo == "hybrid":
        count = metrics["hybrid"]["requests"]
        st.caption(f"All {count} requests remain in each variant's denominator. Latency includes local model execution, IPC and guards; worker initialization is recorded separately.")
        st.dataframe(pd.DataFrame([{"Variant":key,"Correct":f"{round(value['end_to_end_success']*value['requests'])}/{value['requests']}",
                                   "Uses teacher":f"{value['large_model_usage_rate']:.1%}","p95 seconds":round(value['p95_seconds'],3)}
                                  for key,value in metrics.items()]),hide_index=True,width="stretch")
        with st.expander("All quality, rejection, fallback and resource metrics"):
            st.json(metrics)
    elif table is not None:
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.json(metrics)
    if (run / "challenge.json").exists():
        st.markdown("**Independent challenge**")
        challenge = read_json(run / "challenge.json")
        table = scalar_table(challenge)
        if table is not None:
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.json(challenge)
    if (run / "selection.json").exists():
        with st.expander("Validation choice, made before final test evaluation"):
            st.json(read_json(run / "selection.json"))
    report = run / "report.md"
    if report.exists():
        with st.expander("Experiment report", expanded=False):
            st.markdown(report.read_text())
        st.download_button("Download this experiment report", report.read_text(), file_name=f"{demo}-{run.name}.md")

with try_tab:
    if demo == "hybrid":
        st.write("Run a support request through the baseline or hybrid. The pipeline uses real saved models; device commands affect a fresh local simulator.")
        st.caption("API version, resource and permissions are caller-supplied context. They are checked before a tool action. Risk assessment is informational.")
        with st.form("hybrid_prediction"):
            query = st.text_area("Request","Set the thermostat to 22 degrees.")
            variant = st.selectbox("Pipeline",list(metrics),index=list(metrics).index("hybrid"))
            version = st.selectbox("API version",["v1","v2","v3"],index=1)
            resource = st.selectbox("Resource",["device-thermostat","support-ticket","invoice","refund","api-key","incident","role","session-token","payment-method"])
            operation = st.selectbox("API operation",["read","update"],index=1)
            role = st.selectbox("Permission",["operator","viewer"])
            backend = st.selectbox("Query model",["schema","rlvr"])
            row_index = st.number_input("Held-out transaction row",0,999,0)
            metadata = st.text_area("Request metadata vector","[8, 0.02, 5, 4, 400, 0, 1, 6]")
            st.caption("Order: requests/minute, failure ratio, payload KB, distinct paths, account age days, foreign-request flag, path entropy, prior request rate.")
            submitted = st.form_submit_button("Run pipeline")
        if submitted:
            try:
                request = {"id":"interactive","text":query,"version":version,"resource":resource,
                           "operation":operation,"role":role,"query_backend":backend,
                           "transaction_row":int(row_index),"metadata":json.loads(metadata)}
                with tempfile.TemporaryDirectory(prefix="ml-demo-ui-") as temporary:
                    path = Path(temporary) / "request.json"
                    path.write_text(json.dumps(request))
                    observed = execute_command([str(environment_executable("cpu", "ml-demo")),"run","hybrid","--run-dir",str(run),
                                                "--request-json",str(path),"--variant",variant])
                if observed:
                    response = observed["response"]
                    st.markdown(f"**Observed outcome: {response['outcome']}**")
                    if response["reason"]:
                        st.write(response["reason"])
                    st.json(response["result"])
                    st.caption(observed["latency_scope"])
                    st.dataframe(pd.DataFrame([{"stage":s["stage"],"seconds":s["seconds"],"evidence":s["evidence"]}
                                               for s in response["trace"]]),hide_index=True)
                    with st.expander("Stage inputs, outputs, model identities and fallback reasons",expanded=True):
                        st.json(response)
            except (TypeError,ValueError) as error:
                st.error(f"Invalid request context: {error}")
    elif demo == "explainability":
        st.write("Select a held-out transaction and edit its amount. Both predictions and TreeSHAP explanations are computed again from saved weights.")
        with st.form("explanation_edit"):
            row_index = st.number_input("Held-out row",0,999,7)
            amount = st.number_input("Changed amount",0.,100000.,450.)
            submitted = st.form_submit_button("Recompute prediction and explanation")
        if submitted:
            observed = execute_command([str(environment_executable("cpu", "ml-demo")),"run","explainability","--run-dir",str(run),
                                        "--row-index",str(row_index),"--set",f"amount={amount}"])
            if observed:
                st.caption(observed["note"])
                st.dataframe(pd.DataFrame([{"state":key,"calibrated_probability":observed[key]["calibrated_probability"],
                                           "raw_log_odds":observed[key]["raw_log_odds"],
                                           "additivity_error":observed[key]["reconstruction_abs_error"]} for key in ["before","after"]]),hide_index=True)
                st.json(observed)
    elif demo in QUERY_DEFAULTS:
        st.write("This executes the selected implementation locally. Neural variants load saved models; typed/classical alternatives use their documented implementation. It does not replay a stored answer.")
        if demo == "edge":
            st.info("Each request starts a fresh simulator at 20 Celsius with no jobs. A valid schema does not prove that the model understood the request; inspect the before/after states.")
        variants = ["selected", *[key for key in metrics if not key.startswith("fixed_base_pool_")]]
        if demo == "retrieval":
            st.caption("Fixed-candidate reranker comparisons require the frozen evaluation pool and are available in recorded results.")
        with st.form("fresh_prediction"):
            query = st.text_area("Request", QUERY_DEFAULTS[demo])
            variant = st.selectbox("Variant", variants)
            submitted = st.form_submit_button("Run selected variant")
        if submitted:
            command = [str(environment_executable(environments()["demos"][demo], "ml-demo")), "run", demo,
                       "--run-dir", str(run), "--query", query, "--variant", variant]
            if demo in ["retrieval", "rlvr"]:
                command += ["--device", "mps" if demo == "retrieval" else selected.get("device", "cpu")]
            observed = execute_command(command)
            if observed:
                st.json(observed)
    else:
        st.write("This track's interactive model/feature workflow is available through the documented CLI. Its saved predictions and measurements are available in the evidence tab.")

with evidence_tab:
    if st.button("Verify saved artifact checksums"):
        with st.spinner("Verifying recorded files…"):
            try:
                verify_run(run)
                st.success("Every recorded artifact checksum matched.")
            except (ValueError, OSError) as error:
                st.error(str(error))
    files = [name for name in ["failure_examples.json", "predictions.json", "benchmarks.json", "training.json",
                              "grpo_training.json", "sft_control_training.json", "feature_edit.json", "explanations.json",
                              "calibration.json", "knowledge_update.json", "bindings.json", "startup.json", "run.json"]
             if (run / name).exists()]
    filename = st.selectbox("Evidence file", files)
    value = read_json(run / filename)
    if isinstance(value, list):
        st.caption(f"{len(value)} recorded entries")
        if value:
            index = st.number_input("Entry index", 0, len(value)-1, 0, step=1)
            st.json(value[index])
    else:
        st.json(value)
    st.download_button("Download evidence JSON", (run / filename).read_text(), file_name=f"{demo}-{filename}")
