# Local comparison interface

The interface reads the same run artifacts as the CLI. It distinguishes evaluated experiments, smoke runs, failed attempts, independent challenges and fresh execution. All nine tracks have evaluated runs. If a run is missing in another checkout, its view displays that state instead of inventing results.

```bash
uv sync --project environments/ui --frozen
environments/ui/.venv/bin/streamlit run app/main.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Choose a demo and evaluated run. The results tab separates generated test scores from independent challenge scores. The evidence tab exposes failures, predictions, runtime measurements and manifests, and can verify all recorded file hashes.

Fresh inference uses the pinned per-track environment and saved artifacts through the same CLI. Models are loaded on demand; results are not cached-answer replays. D04 starts a fresh simulator for each request and shows its before/after state. The typed parser is identified as a deterministic alternative, not neural generation. D07 lets the user choose a held-out row and change its amount, then recomputes the prediction and contributions. D05/D06 standalone scoring remains available through the documented CLI. D09 accepts explicit API context, permissions and metadata and displays the actual stage trace.

Fresh commands run in their own POSIX process group. If the UI's 180-second deadline expires, the controller and descendant model workers are stopped together. The form checker captures only the new CLI response from that submission and verifies its request, variant and run; saved examples in the evidence tab cannot satisfy it. Smoke comparisons display their actual scenario count.

`environments/ui/.venv/bin/python scripts/check_ui.py` exercises artifact browsing with Streamlit AppTest. The overview also rendered in the in-app browser. `scripts/check_ui_fresh.py` additionally submitted the D01/D02/D04/D08 and D07 forms and checked real local-model responses. D03 and D09 form submissions also passed (`artifacts/validation/ui-fresh-final.json`). The browser displayed all nine evaluated tracks and the complete hybrid comparison. A fresh request to set 23 Celsius produced an actual model command and changed the simulator from 20 to 23; the D07 feature edit was also observed. Screenshots and observed values are recorded in `artifacts/validation/ui-browser.json`.
