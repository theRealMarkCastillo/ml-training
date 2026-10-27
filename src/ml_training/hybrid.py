"""D09: real-model baseline/hybrid orchestration, guards and component ablations."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import time

import numpy as np

from ml_training.common import ROOT, complete_run, digest, environment_executable, mark_failed, new_run, read_json, require_reproduced, verify_run, write_json
from ml_training import hybrid_data as data
from ml_training import hybrid_policy as policy

VARIANTS = ["baseline", "hybrid", "no_screen", "no_router", "no_retrieval", "no_schema",
            "no_rlvr", "no_device", "no_tabular", "no_attribution"]


def bindings(rlvr_run, allow_smoke=False):
    if not (ROOT / "artifacts/validation/selected_cpu_runs.json").is_file():
        raise FileNotFoundError("The capstone binds previously trained component runs, and none are selected yet. "
                                "Run `ml-demo prepare hybrid --auto-select --source-run <D03 run>` to bind the latest "
                                "evaluated runs, or name each run with scripts/select_component_runs.py.")
    cpu = read_json(ROOT / "artifacts/validation/selected_cpu_runs.json")
    structured = read_json(ROOT / "artifacts/validation/selected_structured_runs.json")
    values = {"distillation":read_json(ROOT / "artifacts/validation/selected_distillation_run.json")["run"],
              "schema_dsl":structured["schema_dsl"]["local"], "edge":structured["edge"]["local"],
              "tabular":cpu["tabular-41"], "fast_classifier":cpu["fast_classifier-41"],
              "retrieval":read_json(ROOT / "artifacts/validation/selected_retrieval_run.json")["run"],
              "rlvr":str(rlvr_run)}
    for key, value in values.items():
        path = (ROOT / value).resolve()
        verify_run(path)
        allowed = ["evaluated","smoke"] if allow_smoke else ["evaluated"]
        if read_json(path / "run.json")["status"] not in allowed:
            raise ValueError(f"Capstone requires a full evaluated {key} artifact")
        values[key] = str(path)
    return values


class Worker:
    def __init__(self, kind, config, log_dir, clean=False):
        environment = {"language":"mlx", "retrieval":"transformers", "rlvr":"rlvr"}[kind]
        python = environment_executable(environment, clean=clean)
        self.log = (log_dir / f"{kind}.log").open("w")
        try:
            self.process = subprocess.Popen([str(python),"-m","ml_training.hybrid_workers",kind,str(config)],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                env=os.environ | {"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
        except BaseException:
            self.log.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def call(self, request):
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(request)+"\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=180):
            raise TimeoutError("Local model worker did not answer within 180 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Local worker exited: {self.process.poll()}; inspect its log")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        response["roundtrip_seconds"] = time.perf_counter()-start
        return response

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"task":"close"}\n')
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass  # The child may have exited between poll and write.
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        finally:
            self.selector.close()
            for stream in [self.process.stdin, self.process.stdout]:
                try:
                    stream.close()
                except OSError:
                    pass
            self.log.close()


class Runtime:
    def __init__(self, config, log_dir, clean=False):
        from ml_training.explainability import setup
        from ml_training.fast_classifier import session
        self.bindings = read_json(config)
        _, self.parts, self.tree, self.explainer, self.names, _ = setup(Path(self.bindings["tabular"]))
        self.screen = session(Path(self.bindings["fast_classifier"]) / "model.onnx")
        selection = read_json(Path(self.bindings["fast_classifier"]) / "selection.json")
        self.screen_threshold = selection["validation"][selection["selected_model"]]["threshold"]
        self.policy = None
        self.workers = []
        try:
            for kind in ["language","retrieval","rlvr"]:
                worker = Worker(kind,config,log_dir,clean)
                self.workers.append(worker)
                setattr(self,kind,worker)
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for worker in self.workers:
            try:
                worker.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Local model worker cleanup failed", errors)

    def warm(self):
        records = []
        for task,variant,text in [("route","teacher","I need help with an invoice."),("route","tuned","I need help with an invoice."),
                                   ("query","tuned","Count tickets where status equals open."),
                                   ("device","tuned","Read the thermostat status.")]:
            records.append(self.language.call({"task":task,"variant":variant,"text":text}))
        records.append(self.retrieval.call({"task":"search","variant":"tuned_bi","text":"Inspect the v2 thermostat."}))
        for variant in ["sft","rlvr"]:
            records.append(self.rlvr.call({"task":"query","variant":variant,"text":"Count tickets where status equals open."}))
        return records

    def calibrate(self, rows):
        predictions = [self.language.call({"task":"route","variant":"tuned","text":row["text"]})["result"] for row in rows]
        self.policy = policy.calibrate(rows,predictions)
        return {"policy":self.policy,"predictions":predictions,"rows":rows}

    def execute(self, request, variant):
        from ml_training.fast_classifier import probabilities, validate_request
        from ml_training.tabular import matrix
        from ml_training.explainability import explain_rows
        data.validate_public(request)
        if variant not in VARIANTS:
            raise ValueError("Unknown capstone variant")
        if self.policy is None:
            raise ValueError("Freeze validation escalation policy before execution")
        trace, fallbacks = [], []
        started = time.perf_counter()
        def local_stage(name, inputs, function):
            begin = time.perf_counter()
            try:
                value = function()
            except (ValueError,PermissionError) as error:
                trace.append({"stage":name,"inputs":inputs,"output":{"gate_rejected":True,"error":str(error)},
                              "seconds":time.perf_counter()-begin,"evidence":"actual_runtime_rejection"})
                raise
            trace.append({"stage":name,"inputs":inputs,"output":value,"seconds":time.perf_counter()-begin,
                          "evidence":"actual_local_execution"})
            return value
        def model_stage(name, worker, payload, demo):
            response = worker.call(payload)
            trace.append({"stage":name,"inputs":payload,"output":response["result"],"seconds":response["roundtrip_seconds"],
                          "worker_seconds":response["worker_seconds"],"model_run":self.bindings[demo],
                          "process_peak_rss_bytes_macos":response["process_peak_rss_bytes_macos"],"evidence":"actual_model_execution"})
            return response["result"]
        def finish(outcome, result=None, reason=None):
            return {"id":request["id"],"variant":variant,"outcome":outcome,"result":result,"reason":reason,
                    "fallbacks":fallbacks,"trace":trace,"elapsed_seconds":time.perf_counter()-started,
                    "simulator_only":True}
        metadata = validate_request({"features":[request["metadata"]]})
        learned_screen = variant not in ["baseline","no_screen"]
        def screening():
            probability = float(probabilities(self.screen,metadata)[0]) if learned_screen else None
            return {"model":"D06 ONNX" if learned_screen else "untrained hard rate ceiling",
                    "probability":probability,"threshold":self.screen_threshold if learned_screen else 1000.,
                    "blocked":probability >= self.screen_threshold if learned_screen else bool(metadata[0,0] > 1000)}
        score = local_stage("screen",{"features":request["metadata"]},screening)
        if score["blocked"]:
            return finish("blocked",reason="Request metadata exceeds the frozen screening threshold")
        if request["operation"] == "update" and request["role"] != "operator":
            local_stage("authorization",{"role":request["role"],"operation":request["operation"]},lambda:{"allowed":False})
            return finish("denied",reason="Viewer context cannot authorize an update")
        row_index = request["transaction_row"]
        if row_index >= self.parts["test"].height:
            raise ValueError("Transaction row outside frozen held-out table")
        row = self.parts["test"][row_index:row_index+1]
        learned_risk = variant not in ["baseline","no_tabular"]
        risk = local_stage("risk",{"row_index":row_index,"model":"xgboost" if learned_risk else "fixed_untrained_prior_0.02"},
                           lambda:{"probability":float(self.tree.predict_proba(matrix(row))[0,1]) if learned_risk else .02})
        if learned_risk and variant != "no_attribution":
            local_stage("attribution",{"row_index":row_index},lambda:explain_rows(self.tree,self.explainer,self.names,row)[0][0])
        route_variant = "teacher" if variant in ["baseline","no_router"] else "tuned"
        route = model_stage("route",self.language,{"task":"route","variant":route_variant,"text":request["text"]},"distillation")
        if route_variant == "tuned":
            parsed = route["parsed"]
            why = "invalid routing schema" if parsed is None else "explicit escalation" if parsed["intent"] == "escalate" else "below validation-selected confidence threshold" if route["sequence_confidence"] < self.policy["threshold"] else None
            if why:
                fallbacks.append(why)
                route = model_stage("route_fallback",self.language,{"task":"route","variant":"teacher","text":request["text"]},"distillation")
        if route["parsed"] is None:
            return finish("rejected",reason="Router did not satisfy the runtime schema")
        intent = route["parsed"]["intent"]
        if intent == "escalate":
            question = model_stage("clarify",self.language,{"task":"clarify","variant":"teacher","text":request["text"]},"distillation")
            try:
                return finish("clarify",policy.checked_clarification(question["text"]))
            except ValueError as error:
                return finish("rejected",reason=str(error))
        retrieval_variant = "bm25" if variant == "baseline" else "base_bi" if variant == "no_retrieval" else "tuned_bi"
        search_text = f"{request['text']} Client API {request['version']}, resource {request['resource']}, {request['operation']} operation."
        found = model_stage("retrieve",self.retrieval,{"task":"search","variant":retrieval_variant,"text":search_text},"retrieval")
        doc = found["top_five"][0]
        if not policy.document_guard(doc,request):
            return finish("rejected",reason="Retrieved document conflicts with the explicit resource, version or operation")
        if intent in ["billing","access","incident"]:
            return finish("reference",policy.reference(doc))
        if intent == "device":
            generator = "teacher" if variant in ["baseline","no_device"] else "tuned"
            output = model_stage("device_generate",self.language,{"task":"device","variant":generator,"text":request["text"]},"edge")
            try:
                result = local_stage("device_dispatch",{"text":output["text"],"role":request["role"]},lambda:policy.checked_device(output["text"],request))
            except PermissionError as error:
                return finish("denied",reason=str(error))
            except ValueError:
                if generator == "teacher":
                    return finish("rejected",reason="Teacher device output failed the runtime gate")
                fallbacks.append("invalid device payload")
                output = model_stage("device_fallback",self.language,{"task":"device","variant":"teacher","text":request["text"]},"edge")
                try:
                    result = local_stage("device_dispatch",{"text":output["text"],"role":request["role"]},lambda:policy.checked_device(output["text"],request))
                except (ValueError,PermissionError) as error:
                    return finish("rejected",reason=str(error))
            return finish("device",result)
        if intent == "query":
            teacher = variant == "baseline" or (variant == "no_schema" and request["query_backend"] == "schema")
            if teacher:
                output = model_stage("query_generate",self.language,{"task":"query","variant":"teacher","text":request["text"]},"schema_dsl")
            elif request["query_backend"] == "schema":
                output = model_stage("query_generate",self.language,{"task":"query","variant":"tuned","text":request["text"]},"schema_dsl")
            else:
                output = model_stage("query_generate",self.rlvr,{"task":"query","variant":"sft" if variant == "no_rlvr" else "rlvr","text":request["text"]},"rlvr")
            try:
                result = local_stage("query_execute",{"text":output["text"]},lambda:policy.checked_query(output["text"],request["text"]))
            except ValueError:
                if teacher:
                    return finish("rejected",reason="Teacher query failed the bounded schema/public verifier")
                fallbacks.append("query schema or public semantic check failed")
                output = model_stage("query_fallback",self.language,{"task":"query","variant":"teacher","text":request["text"]},"schema_dsl")
                try:
                    result = local_stage("query_execute",{"text":output["text"]},lambda:policy.checked_query(output["text"],request["text"]))
                except ValueError as error:
                    return finish("rejected",reason=str(error))
            return finish("query",result)
        return finish("rejected",reason="Unsupported route")


def grade(scenario, result):
    from ml_training.query_dsl import equivalent
    from ml_training.schema_data import tables
    expected = scenario["expected"]
    correct = result["outcome"] == expected["outcome"]
    if correct and expected["outcome"] == "reference":
        correct = result["result"]["doc_id"] == expected["doc_id"]
    elif correct and expected["outcome"] == "device":
        correct = result["result"]["command"] == expected["value"]
    elif correct and expected["outcome"] == "query":
        correct = equivalent(json.dumps(result["result"]["ast"]),expected["value"],tables(expected["value"],expected["fixture_seed"],hidden=True))["passed"]
    retrieval = [s for s in result["trace"] if s["stage"] == "retrieve"]
    wrong_version = bool(retrieval and retrieval[0]["output"]["top_five"][0]["version"] != scenario["public"]["version"])
    legitimate = expected["outcome"] not in ["blocked","denied"] and not (expected["outcome"] == "device" and expected["value"]["action"] == "reject")
    return {"correct":bool(correct),"wrong_version_retrieval":wrong_version,
            "false_rejection":legitimate and result["outcome"] in ["blocked","denied","rejected"],
            "expected":expected}


def summarize(rows):
    times = [r["elapsed_seconds"] for r in rows]
    risk = [(r["risk_label"], s["output"]["probability"]) for r in rows for s in r["trace"] if s["stage"] == "risk"]
    return {"requests":len(rows),"end_to_end_success":float(np.mean([r["grade"]["correct"] for r in rows])),
            "false_rejections_over_all_requests":float(np.mean([r["grade"]["false_rejection"] for r in rows])),
            "fallback_rate":float(np.mean([bool(r["fallbacks"]) for r in rows])),
            "wrong_version_retrieval_over_all_requests":float(np.mean([r["grade"]["wrong_version_retrieval"] for r in rows])),
            "p50_seconds":float(np.percentile(times,50)),"p95_seconds":float(np.percentile(times,95)),
            "p99_seconds":float(np.percentile(times,99)),"concurrency":1,
            "attribution_coverage_over_all_requests":float(np.mean([any(s["stage"] == "attribution" for s in r["trace"]) for r in rows])),
            "risk_scored_requests":len(risk),"risk_brier_on_scored_requests":float(np.mean([(y-p)**2 for y,p in risk])) if risk else None,
            "risk_positive_labels":sum(y for y,p in risk),
            "large_model_usage_rate":float(np.mean([any(s["inputs"].get("variant")=="teacher" for s in r["trace"]) for r in rows])),
            "generation_tokens":sum(s["output"].get("generation_tokens",0) for r in rows for s in r["trace"]),
            "runtime_gate_rejections":sum(bool(s["output"].get("gate_rejected")) for r in rows for s in r["trace"]),
            "stage_seconds_total":{stage:sum(s["seconds"] for r in rows for s in r["trace"] if s["stage"]==stage)
                                   for stage in sorted({s["stage"] for r in rows for s in r["trace"]})},
            "worker_peak_rss_bytes_max_by_component":{component:max(s["process_peak_rss_bytes_macos"] for r in rows for s in r["trace"] if s.get("model_run")==component)
                for component in sorted({s["model_run"] for r in rows for s in r["trace"] if "model_run" in s})},
            "invalid_payload_or_semantic_check_fallbacks":sum(any("payload" in f or "semantic" in f for f in r["fallbacks"]) for r in rows)}


def run(rlvr_run, profile="local", seed=41):
    folder = data.prepare()
    scenarios, calibration = data.load(folder)
    if profile == "smoke":
        scenarios = [scenarios[i] for i in [0,8,14,16,17,23,26]]
    refs = bindings(rlvr_run,allow_smoke=profile=="smoke")
    run = new_run("hybrid",profile,seed)
    component_hashes_before = {key:digest(Path(value)/"run.json") for key,value in refs.items()}
    write_json(run / "bindings.json",refs)
    print(f"Run directory: {run}",flush=True)
    runtime = None
    try:
        start = time.perf_counter()
        runtime = Runtime(run / "bindings.json",run)
        warmup = runtime.warm()
        write_json(run / "startup.json",{"seconds":time.perf_counter()-start,"warmups":warmup,
                                         "scope":"All shared workers/model caches initialized; not a per-variant memory estimate"})
        write_json(run / "calibration.json",runtime.calibrate(calibration))
        write_json(run / "selection.json",{"escalation_policy":runtime.policy,"screen_threshold":runtime.screen_threshold,
                                           "frozen_before_test":True,"runtime_variants":VARIANTS})
        metrics, predictions = {}, []
        for variant in VARIANTS:
            observed = []
            for scenario in scenarios:
                result = runtime.execute(scenario["public"],variant)
                result["grade"] = grade(scenario,result)
                result["risk_label"] = int(runtime.parts["test"]["label"][scenario["public"]["transaction_row"]])
                observed.append(result)
                with (run / "traces.jsonl").open("a") as stream:
                    stream.write(json.dumps(result)+"\n")
            metrics[variant] = summarize(observed)
            predictions.extend(observed)
            write_json(run / "metrics.json",metrics)
            print(f"D09 {variant} success={metrics[variant]['end_to_end_success']:.4f} fallback={metrics[variant]['fallback_rate']:.4f}",flush=True)
        update = knowledge_update(runtime,scenarios[0]["public"])
        component_hashes_after = {key:digest(Path(value)/"run.json") for key,value in refs.items()}
        if component_hashes_before != component_hashes_after:
            raise AssertionError("Component artifact manifests changed during integration")
        write_json(run / "knowledge_update.json",{**update,
            "model_manifest_hashes_unchanged":True,"model_manifests_before":component_hashes_before,"model_manifests_after":component_hashes_after,
            "note":"Actual corpus edit and embedding reindex; returned facts are parsed from retrieved text. This is not evidence that weights learned new knowledge."})
        write_json(run / "predictions.json",predictions)
        write_json(run / "failure_examples.json",[r for r in predictions if not r["grade"]["correct"]])
        write_report(run,metrics)
        runtime.close()
        runtime = None
        for log in run.glob("*.log"):
            if "Impacting Interactivity" in log.read_text():
                raise RuntimeError("Worker GPU execution error makes this experiment unreliable")
        complete_run(run,dataset_path=str(folder.relative_to(ROOT)),dataset_sha256=digest(folder/"manifest.json"),
            component_manifests={key:{"run":value,"sha256":digest(Path(value)/"run.json")} for key,value in refs.items()},
            synthetic=True,scenario_count=len(scenarios),variants=VARIANTS,actual_model_execution=True,
            teacher_api_charge_usd=0,latency_scope="Warm persistent workers; actual sequential end-to-end requests including IPC and gates; startup separate")
    except BaseException as error:
        mark_failed(run, error)
        raise
    finally:
        if runtime:
            runtime.close()
    return run


def write_report(run,metrics):
    lines=["# D09: Actual baseline and hybrid support workflows","", "| Variant | Task success | False rejection/all | Fallback | Warm p95 seconds |", "| --- | --- | --- | --- | --- |"]
    for name,m in metrics.items():
        lines.append(f"| {name} | {m['end_to_end_success']:.4f} | {m['false_rejections_over_all_requests']:.4f} | {m['fallback_rate']:.4f} | {m['p95_seconds']:.3f} |")
    lines += ["", f"Every variant executes every one of the same {metrics['hybrid']['requests']} frozen synthetic requests. No escalation or rejection is removed from the task-success denominator. "
              "The no-training agent baseline uses a prompted local teacher, BM25, deterministic guards, a hard rate ceiling and a prior-risk comparator. "
              "The hybrid uses D06 screening, D01 routing with validation-calibrated escalation, D08 retrieval, D02/D03 query generation, D04 simulated commands, and D05/D07 risk/explanations. "
              "Ablations replace one component with its stated simpler or untuned alternative. Disabling feature-based risk makes its feature attribution inapplicable; the attribution-only ablation leaves risk predictions intact.",
              "", "Risk assessment is informational, so its Brier score and attribution coverage are reported separately from the primary support-task outcome. "
              "Runtime receives only public request/context fields; expected outcomes and hidden semantic fixtures are used only by the evaluator. "
              "Clarification success checks the escalation branch and bounded question schema, not human-rated usefulness. "
              "Schema-valid requests can still be misunderstood. Public typed-query checks cover their declared grammar only. Device actions affect a fresh simulator, never a physical device.",
              "", "Each trace preserves stage input/output, model artifact identity, actual timing and fallback reason. Large-model usage includes clarification and repair calls; baseline teacher usage is not labeled fallback. "
              "Warm latency includes IPC and validation but excludes separately measured worker initialization. Shared worker peak memory is cumulative, not a per-variant footprint. "
              "No hosted API was used and energy/depreciation costs were not measured. These small synthetic samples do not establish production reliability or precise tail latency."]
    (run/"report.md").write_text("\n".join(lines)+"\n")


def knowledge_update(runtime, public):
    """Use the actual pipeline's top document and final response before/after."""
    request = dict(public,id="knowledge-update",version="v2")
    before = runtime.execute(request,"hybrid")
    if before["outcome"] != "reference" or before["result"]["doc_id"] != "invoice-v2-read":
        raise AssertionError("Knowledge update probe did not retrieve the intended top reference")
    doc = next(s["output"]["top_five"][0] for s in before["trace"] if s["stage"]=="retrieve")
    changed = doc["text"].replace("Responses contain result and request_id.","Responses contain payload and request_id.")
    if changed == doc["text"]:
        raise AssertionError("Document update did not change the intended field")
    update = runtime.retrieval.call({"task":"update_document","doc_id":doc["doc_id"],"text":changed})["result"]
    after = runtime.execute(request,"hybrid")
    if after["outcome"] != "reference" or after["result"]["response_field"] != "payload" or before["result"]["response_field"] != "result":
        raise AssertionError("Updated document fact did not flow through the actual pipeline")
    return {"request":request,"before":before,"after":after,"update":update}


def execution_content(value):
    """Replay checks content, decisions and rankings, not nondeterministic timing."""
    ignored = {"seconds","worker_seconds","elapsed_seconds","process_peak_rss_bytes_macos",
               "generation_seconds","batch_elapsed_seconds"}
    if isinstance(value,dict):
        return {key:execution_content(item) for key,item in value.items() if key not in ignored}
    if isinstance(value,list):
        return [execution_content(item) for item in value]
    return value


def reevaluate(run, output_dir, clean=False, tolerance=0.):
    verify_run(run)
    info = read_json(run / "run.json")
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("Capstone dataset manifest changed")
    scenarios, calibration = data.load(folder)
    if info["profile"] == "smoke":
        scenarios = [scenarios[i] for i in [0,8,14,16,17,23,26]]
    verify_components(info)
    output_dir.mkdir(parents=True,exist_ok=False)
    runtime = Runtime(run / "bindings.json",output_dir,clean)
    try:
        warmup = runtime.warm()
        selected = runtime.calibrate(calibration)
        require_reproduced(execution_content(selected), execution_content(read_json(run / "calibration.json")),
                           "D09 routing calibration", tolerance)
        original = {(r["variant"],r["id"]):r for r in read_json(run / "predictions.json")}
        count = 0
        for variant in info["variants"]:
            for scenario in scenarios:
                result = runtime.execute(scenario["public"],variant)
                result["grade"] = grade(scenario,result)
                result["risk_label"] = int(runtime.parts["test"]["label"][scenario["public"]["transaction_row"]])
                with (output_dir / "traces.jsonl").open("a") as stream:
                    stream.write(json.dumps(result)+"\n")
                require_reproduced(execution_content(result), execution_content(original[(variant,result["id"])]),
                                   f"D09 {variant}/{result['id']} pipeline output", tolerance)
                count += 1
            print(f"Reproduced D09 {variant}",flush=True)
        update = knowledge_update(runtime,scenarios[0]["public"])
        saved_update = read_json(run / "knowledge_update.json")
        require_reproduced(execution_content(update), execution_content({key:saved_update[key] for key in update}),
                           "D09 document update", tolerance)
        write_json(output_dir / "knowledge_update.json",update)
        result = {"status":"reproduced","run":str(run),"clean_worker_environments":clean,
                  "actual_executions":count,"exact_content_and_decisions":True,
                  "scope":"All recorded traces except nondeterministic time/RSS; includes raw generated text, rankings, probabilities, grades and document update"}
        write_json(output_dir / "result.json",result)
        return result
    finally:
        runtime.close()


def infer_request(run, request, variant="hybrid"):
    import tempfile
    verify_run(run)
    verify_components(read_json(run / "run.json"))
    data.validate_public(request)
    with tempfile.TemporaryDirectory(prefix="ml-demo-hybrid-") as tmp:
        runtime = Runtime(run / "bindings.json",Path(tmp))
        try:
            runtime.policy = read_json(run / "selection.json")["escalation_policy"]
            result = runtime.execute(request,"hybrid" if variant=="selected" else variant)
            return {"execution":"fresh_local_models","request":request,"response":result,
                    "latency_scope":"Fresh worker/model loads included; incomparable to recorded warm requests"}
        finally:
            runtime.close()


def verify_components(info):
    for entry in info["component_manifests"].values():
        verify_run(Path(entry["run"]))
        if digest(Path(entry["run"]) / "run.json") != entry["sha256"]:
            raise ValueError("A capstone component manifest changed")
