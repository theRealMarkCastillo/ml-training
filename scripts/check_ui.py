"""Exercise artifact browsing without pretending to run model inference."""

from streamlit.testing.v1 import AppTest
from ml_training.common import ROOT, write_json

app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run()
assert not app.exception, app.exception
assert app.title[0].value == "ML Training Demo Lab"
visited = []
for demo in ["distillation", "schema_dsl", "rlvr", "edge", "tabular", "fast_classifier", "explainability", "retrieval", "hybrid"]:
    app.sidebar.selectbox[0].select(demo).run()
    assert not app.exception, (demo, app.exception)
    visited.append(demo)
write_json(ROOT / "artifacts/validation/ui.json", {"status": "artifact_browsing_passed", "demos": visited,
           "scope": "Streamlit AppTest; no browser rendering or fresh model-inference button claim"})
print("UI artifact browsing checks passed")
