"""Teaching-evidence regressions; no local models or original artifacts required."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'notebooks/data/recorded_results.json'


def read():
    return json.loads(SNAPSHOT.read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_snapshot_is_portable_and_has_explicit_provenance():
    data = read()
    assert set(data['demos']) == {f'D{i:02}' for i in range(1, 10)}
    assert data['evidence_kind'] == 'recorded_local_experiments'
    assert '/Users/' not in SNAPSHOT.read_text()
    for path, sha in data['sources'].items():
        assert not Path(path).is_absolute() and '..' not in Path(path).parts
        assert len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)


def test_structured_gate_denominators_are_consistent():
    d = read()['demos']['D02']
    for population in ['metrics', 'challenge']:
        for metrics in d[population].values():
            assert 0 <= metrics['joint_accuracy'] <= metrics['schema_validity'] <= metrics['json_parse_rate'] <= 1
            assert metrics['rows'] > 0
            assert math.isclose(metrics['joint_accuracy'] * metrics['rows'],
                                round(metrics['joint_accuracy'] * metrics['rows']))


def test_reward_groups_are_counted_without_dropping_constant_groups():
    t = read()['demos']['D03']['training']
    assert t['nonconstant_reward_groups'] + t['constant_reward_groups'] == t['optimizer_steps']


def test_reranking_metrics_recompute_from_all_exported_queries():
    d = read()['demos']['D08']
    populations = []
    for variant, metrics in d['metrics'].items():
        rows = [r for r in d['ranks'] if r['variant'] == variant]
        ids = {r['query_id'] for r in rows}
        assert len(rows) == len(ids) == metrics['queries']
        populations.append(ids)
        assert all(isinstance(r['rank'], int) and r['rank'] >= 1 for r in rows)
        assert math.isclose(sum(1/r['rank'] if r['rank'] <= 10 else 0 for r in rows)/len(rows), metrics['mrr_at_10'])
        assert math.isclose(sum(r['rank'] == 1 for r in rows)/len(rows), metrics['top1_accuracy'])
    assert all(ids == populations[0] for ids in populations)


def test_explanation_reconstructs_raw_score_not_calibrated_probability():
    d = read()['demos']['D07']
    for row in [d['feature_edit']['before'], d['feature_edit']['after']]:
        raw = row['base_log_odds'] + sum(row['contributions_log_odds'].values())
        assert abs(raw - row['raw_log_odds']) < 1e-5
        assert abs(1 / (1 + math.exp(-raw)) - row['uncalibrated_probability']) < 1e-6
    variants = d['correlated_features']['variants']
    assert variants[0]['probabilities'] == variants[1]['probabilities']


def test_hybrid_outcome_counts_reconstruct_aggregate_success():
    d = read()['demos']['D09']
    for variant, categories in d['outcomes'].items():
        total = sum(r['total'] for r in categories.values())
        correct = sum(r['correct'] for r in categories.values())
        assert total == d['metrics'][variant]['requests']
        assert correct / total == d['metrics'][variant]['end_to_end_success']
    cohort = d['common_risk_cohort']
    assert all(r['rows'] == len(cohort['ids']) for r in cohort['variants'].values())


def test_latency_units_and_profile_sources_are_not_mixed():
    data = read()
    for r in data['demos']['D06']['paired_profiles']:
        assert r['source_manifest_sha256'] == data['sources'][r['run'] + '/run.json']
        for batch, profiles in r['batch_profiles'].items():
            for backend in profiles['backends'].values():
                assert backend['unit'] == 'ms'
                assert backend['batch_size'] == int(batch)
                assert math.isclose(backend['rows_per_second'], int(batch) * 1000 / backend['mean'])


def test_custom_snapshot_does_not_silently_fall_back(tmp_path):
    pytest.importorskip('matplotlib')
    lab = module(ROOT / 'notebooks/lab.py', 'notebook_lab_test')
    with pytest.raises(FileNotFoundError):
        lab.load_demo('D01', tmp_path / 'missing.json')
    invalid = tmp_path / 'invalid.json'
    invalid.write_text('{"schema_version": 999}')
    with pytest.raises(ValueError):
        lab.load_demo('D01', invalid)
    valid = tmp_path / 'custom.json'
    data = read()
    data['demos']['D01']['metrics']['teacher']['joint_accuracy'] = .25
    valid.write_text(json.dumps(data))
    assert lab.load_demo('D01', valid)['metrics']['teacher']['joint_accuracy'] == .25


def test_changed_cells_cannot_reuse_saved_execution():
    nbformat = pytest.importorskip('nbformat')
    pytest.importorskip('nbclient')
    checks = module(ROOT / 'scripts/check_notebooks.py', 'notebook_checks_test')
    path = ROOT / 'notebooks/D01_distillation.ipynb'
    book = nbformat.read(path, as_version=4)
    assert checks.check_outputs(path, book) >= 2
    changed = copy.deepcopy(book)
    changed.cells[2].source += '\nrequest_volume = 999'
    with pytest.raises(ValueError, match='stale output'):
        checks.check_outputs(path, changed)


def test_teaching_paragraphs_do_not_render_as_indented_code():
    pytest.importorskip('nbconvert')
    from nbconvert.filters.markdown_mistune import markdown2html_mistune
    for path in (ROOT / 'notebooks').glob('D*.ipynb'):
        book = json.loads(path.read_text())
        for cell in book['cells']:
            if cell['cell_type'] == 'markdown':
                # Lessons put runnable examples in code cells; prose stays prose.
                assert '<pre><code>' not in markdown2html_mistune(''.join(cell['source'])), path.name


@pytest.mark.parametrize('break_even', [None, 0.0, -1.0])
def test_amortization_cell_handles_no_positive_break_even(break_even, capsys):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    book = json.loads((ROOT / 'notebooks/D01_distillation.ipynb').read_text())
    cell = next(''.join(c['source']) for c in book['cells']
                if c['cell_type'] == 'code' and 'fixed_seconds' in ''.join(c['source']))
    demo = read()['demos']['D01']
    demo['costs']['time_only_break_even_requests'] = break_even
    plt.close('all')
    exec(compile(cell, 'D01 amortization cell', 'exec'), {'d': demo, 'np': np, 'plt': plt})
    assert not plt.get_fignums()
    assert 'No positive time-only break-even' in capsys.readouterr().out


@pytest.mark.parametrize('absolute_profile', [False, True])
@pytest.mark.parametrize('absolute_selection', [False, True])
@pytest.mark.parametrize('profile_case', ['same', 'wrong_run', 'wrong_hash'])
def test_export_accepts_equivalent_run_paths(tmp_path, monkeypatch, absolute_profile, absolute_selection, profile_case):
    exporter = module(ROOT / 'scripts/export_notebook_snapshot.py', 'notebook_export_test')
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    records = {}

    def add(path, value):
        records[(tmp_path / path).resolve()] = value

    def selector_run(path):
        return str(tmp_path / path) if absolute_selection else path

    def profile_run(path):
        return str(tmp_path / path) if absolute_profile else path

    for demo in ['distillation', 'schema_dsl', 'rlvr', 'edge', 'tabular',
                 'fast_classifier', 'explainability', 'retrieval', 'hybrid']:
        folder = f'artifacts/{demo}/sample'
        add(folder + '/run.json', {'status': 'evaluated', 'started_at': 'recorded',
                                  'seed': 41, 'environment': {}})
        for name in ['metrics', 'challenge', 'costs', 'selection', 'quantization',
                     'benchmarks', 'feature_edit', 'correlated_features']:
            add(folder + '/' + name + '.json', {})
        add(folder + '/predictions.json', [])
        if demo in ['distillation', 'rlvr', 'retrieval', 'hybrid']:
            add(f'selections/selected_{demo}_run.json', {'run': selector_run(folder)})
    add('selections/selected_structured_runs.json', {
        demo: {'local': selector_run(f'artifacts/{demo}/sample')} for demo in ['schema_dsl', 'edge']})
    add('selections/selected_cpu_runs.json', {
        f'{demo}-41': selector_run(f'artifacts/{demo}/sample')
        for demo in ['tabular', 'fast_classifier', 'explainability']})
    add('artifacts/rlvr/sample/grpo_training.json', {key: 1 for key in [
        'optimizer_steps', 'nonconstant_reward_groups', 'constant_reward_groups',
        'parameters_changed', 'elapsed_training_seconds']})
    import hashlib
    manifest_bytes = json.dumps(records[tmp_path / 'artifacts/fast_classifier/sample/run.json']).encode()
    add('selections/classifier-batches.json', {'runs': [{
        'run': profile_run('artifacts/fast_classifier/other' if profile_case == 'wrong_run'
                           else 'artifacts/fast_classifier/sample'),
        'source_manifest_sha256': '0' * 64 if profile_case == 'wrong_hash'
                                 else hashlib.sha256(manifest_bytes).hexdigest()}]})
    add('selections/hybrid-analysis.json', {'run': profile_run('artifacts/hybrid/sample'),
                                           'common_risk_cohort': {}})
    monkeypatch.setattr(Path, 'read_bytes', lambda path: json.dumps(records[path.resolve()]).encode())
    if profile_case != 'same':
        with pytest.raises(ValueError, match='do not match' if profile_case == 'wrong_run' else 'manifest changed'):
            exporter.export_snapshot(tmp_path / 'selections')
        return
    snapshot = exporter.export_snapshot(tmp_path / 'selections')
    assert snapshot['demos']['D06']['paired_profiles'][0]['run'] == 'artifacts/fast_classifier/sample'
    assert snapshot['demos']['D09']['run'] == 'artifacts/hybrid/sample'
    assert str(tmp_path) not in json.dumps(snapshot)
