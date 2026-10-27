"""Check the selected evidence bundle; incomplete work stays a failed check."""

from pathlib import Path
import subprocess
import sys

from ml_training.common import ROOT, digest, environment_executable, read_json, verify_run, write_json
from ml_training.ui_evidence import validate_submission

validation = ROOT / "artifacts/validation"
checks = []


def check(name, function):
    try:
        details = function()
        checks.append({"check":name,"passed":True,"details":details})
    except (AssertionError,ValueError,OSError,KeyError,StopIteration) as error:
        checks.append({"check":name,"passed":False,"error":f"{type(error).__name__}: {error}"})


def selected():
    cpu = read_json(validation / "selected_cpu_runs.json")
    structured = read_json(validation / "selected_structured_runs.json")
    return {"distillation":read_json(validation / "selected_distillation_run.json")["run"],
            "schema_dsl":structured["schema_dsl"]["local"],"edge":structured["edge"]["local"],
            "rlvr":read_json(validation / "selected_rlvr_run.json")["run"],
            "retrieval":read_json(validation / "selected_retrieval_run.json")["run"],
            "hybrid":read_json(validation / "selected_hybrid_run.json")["run"],
            **{key:cpu[f"{key}-41"] for key in ["tabular","fast_classifier","explainability"]}}


def artifacts():
    refs = selected()
    for demo,path in refs.items():
        run = ROOT / path
        verify_run(run)
        info = read_json(run / "run.json")
        assert info["status"]=="evaluated" and info["profile"]=="local",demo
        if "dataset_path" in info:
            assert digest(ROOT / info["dataset_path"] / "manifest.json")==info["dataset_sha256"],demo
        if demo != "explainability":
            assert (run / "metrics.json").is_file() and (run / "failure_examples.json").is_file(),demo
        if demo in ["distillation","schema_dsl","edge"]:
            training = read_json(run / "training.json")
            assert training["optimizer_steps_executed"]>0
            assert training["initial_adapter_sha256"]!=training["selected_adapter_sha256"]
        if demo=="retrieval":
            for name in ["bi_encoder","cross_encoder"]:
                training = read_json(run / f"{name}_training.json")
                assert training["optimizer_steps_executed"]>0
                assert training["before_parameter_sha256"]!=training["selected_after_parameter_sha256"]
        if demo=="rlvr":
            training = read_json(run / "grpo_training.json")
            control = read_json(run / "sft_control_training.json")
            assert training["optimizer_steps"]==50 and training["parameters_changed"]
            assert training["nonconstant_reward_groups"]>0
            assert training["initial_adapter_sha256"]==control["initial_adapter_sha256"]
        if demo=="explainability":
            assert info["max_additivity_residual"]<=info["tolerance"]
    return refs


def reproductions():
    refs = {name:str((ROOT/path).resolve()) for name,path in selected().items()}
    cpu = read_json(validation / "cpu-with-http.json")
    assert all(row["status"]=="reproduced" for row in cpu["evaluations"])
    assert cpu["http"]["status"]=="passed" and len(cpu["http"]["cases"])==6
    assert all(row["expected_status"]==row["actual_status"] for row in cpu["http"]["cases"])
    assert {refs[name] for name in ["tabular","fast_classifier","explainability"]}.issubset({r["run"] for r in cpu["evaluations"]})
    ret = read_json(validation / "retrieval.json")["reevaluation"]
    assert ret["status"]=="reproduced" and str(Path(ret["run"]).resolve())==refs["retrieval"]
    slm = read_json(validation / "slm.json")["completed_runs"]
    assert all(row["status"]=="reproduced" for row in slm)
    assert {refs[name] for name in ["distillation","schema_dsl","edge"]}=={str(Path(r["run"]).resolve()) for r in slm}
    for name in ["rlvr","hybrid"]:
        replay = read_json(validation / f"{name}.json")["result"]
        assert replay["status"]=="reproduced" and str(Path(replay["run"]).resolve())==refs[name]
    offline = read_json(validation / "edge-offline.json")
    assert offline["network_denial_observed"]["errno"]==1 and offline["simulator_only"]
    return {"tracks":9,"scope":"Independent locked-environment artifact reload and inference, not retraining replication"}


def hybrid_coverage():
    run = ROOT / selected()["hybrid"]
    predictions = read_json(run / "predictions.json")
    metrics = read_json(run / "metrics.json")
    assert len(metrics)==10 and len(predictions)==320
    assert all(value["requests"]==32 for value in metrics.values())
    ids = [{row["id"] for row in predictions if row["variant"]==variant} for variant in metrics]
    assert all(group==ids[0] and len(group)==32 for group in ids)
    hybrid = [row for row in predictions if row["variant"]=="hybrid"]
    stages = {stage["stage"] for row in hybrid for stage in row["trace"]}
    required = {"screen","risk","attribution","route","retrieve","query_generate","query_execute","device_generate","device_dispatch","authorization","route_fallback","clarify"}
    assert required.issubset(stages),f"Unexercised branches: {required-stages}"
    assert any(row["outcome"]=="blocked" for row in hybrid)
    assert any(row["fallbacks"] for row in hybrid)
    update = read_json(run / "knowledge_update.json")
    assert update["before"]["result"]["response_field"]=="result" and update["after"]["result"]["response_field"]=="payload"
    assert update["model_manifest_hashes_unchanged"] and not update["update"]["retrained"]
    return {"actual_requests":320,"stages":sorted(stages),"knowledge_update":"actual regenerated response"}


def ui():
    recorded = read_json(validation / "ui.json")
    assert set(recorded["demos"])==set(selected())
    actual = [*read_json(validation / "ui-fresh-components.json")["completed"],
              *read_json(validation / "ui-fresh-final.json")["completed"]]
    assert {row["demo"] for row in actual}=={"distillation","schema_dsl","rlvr","edge","retrieval","explainability","hybrid"}
    for row in actual:
        assert row["execution_source"]=="submitted_cli"
        validate_submission(row,row["demo"],row["submitted"],row["variant"],row["run_name"])
    browser = read_json(validation / "ui-browser.json")
    assert browser["status"]=="browser_verification_passed"
    return {"artifact_views":9,"fresh_forms":7,"browser":browser}


def tests():
    result = subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    selection = subprocess.run([str(environment_executable("rlvr", clean=True)),
                                str(ROOT / "scripts/check_rlvr_selection.py")],cwd=ROOT,capture_output=True,text=True)
    assert selection.returncode==0,selection.stdout+selection.stderr
    return {"stdout":result.stdout,"stderr":result.stderr,"python":sys.executable,
            "public_only_selection_contract":selection.stdout}


def resource_profile():
    profile = read_json(validation / "rlvr-performance.json")
    assert (ROOT / profile["run"]).resolve()==(ROOT / selected()["rlvr"]).resolve()
    assert set(profile["variants"])=={"base","sft","rlvr","sft_control"}
    for variant,value in profile["variants"].items():
        assert value["samples"]==12 and value["warmups"]==1 and value["batch_size"]==1
        assert value["cold"]["actual_output"]["generation_tokens"]>0,variant
    batches = read_json(validation / "classifier-batches.json")
    assert batches["status"]=="measured"
    assert {row["seed"] for row in batches["runs"]}=={41,42,43}
    cpu = read_json(validation / "selected_cpu_runs.json")
    assert {str((ROOT / row["run"]).resolve()) for row in batches["runs"]}=={
        str((ROOT / cpu[f"fast_classifier-{seed}"]).resolve()) for seed in [41,42,43]}
    for row in batches["runs"]:
        assert digest(ROOT / row["run"] / "run.json")==row["source_manifest_sha256"]
        assert set(row["batch_profiles"])=={"1","256"}
        for size,pair in row["batch_profiles"].items():
            assert pair["parity_max_abs_error"]<=1e-5
            assert set(pair["backends"])=={"native","onnx"}
            assert all(timing["batch_size"]==int(size) and timing["samples"]==200 and timing["concurrency"]==1
                       for timing in pair["backends"].values())
    return {"rlvr_warm_and_fresh_process":"passed","native_and_onnx_batch_1_and_256":"passed for three seeds",
            "other_profiles":"Recorded in the CPU, retrieval and SLM reproduction bundles"}


def explanation_stability():
    result = read_json(validation / "explanation-stability.json")
    assert result==read_json(validation / "explanation-stability-reproduced.json")
    assert result["rows_considered"]==100 and result["factors"]==[.99,1.01]
    assert result["numeric_edits"]==2*result["rows_measured"]
    assert result["summary"]["max_margin_delta_reconstruction_error"]<=2e-5
    assert str((ROOT / result["run"]).resolve())==str((ROOT / selected()["explainability"]).resolve())
    source = ROOT / read_json(ROOT / result["run"] / "run.json")["source_run"]
    assert digest(source / "run.json")==result["source_manifest_sha256"]
    return {"numeric_edits":result["numeric_edits"],"exact_clean_replay":True,"summary":result["summary"]}


def documents():
    names = ["project/IMPLEMENTATION_PLAN.md","CPU_DEMOS.md","CPU_RESULTS.md","RETRIEVAL_DEMO.md","RETRIEVAL_RESULTS.md",
             "SLM_DEMOS.md","SLM_RESULTS.md","RLVR_DEMO.md","RLVR_RESULTS.md","HYBRID_DEMO.md","HYBRID_RESULTS.md",
             "COMPARISON_UI.md","project/FINAL_REPORT.md","REPRODUCTION.md","project/CONFORMANCE.md"]
    for name in names:
        assert (ROOT / "docs" / name).is_file(),name
    return names


for name,fn in [("selected_artifacts_and_training",artifacts),("all_track_reproductions",reproductions),
                ("hybrid_actual_branch_coverage",hybrid_coverage),("comparison_ui",ui),
                ("contract_tests",tests),("resource_profile",resource_profile),("small_perturbation_stability",explanation_stability),
                ("walkthroughs_and_final_reports",documents)]:
    check(name,fn)
result = {"status":"passed" if all(c["passed"] for c in checks) else "incomplete","checks":checks,
          "limitation":"Evidence audit does not establish production suitability or independent scientific replication."}
write_json(validation / "final-audit.json",result)
print(result["status"])
for item in checks:
    print(item["check"],"PASS" if item["passed"] else item["error"])
raise SystemExit(0 if result["status"]=="passed" else 1)
