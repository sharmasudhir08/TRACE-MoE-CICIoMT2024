"""Checks of run aggregation:  python -m pytest -q tests"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_iomt.compare import latest_runs  # noqa: E402


def test_latest_runs_keeps_only_the_newest_copy():
    runs = [{"_dir": "results/main/rf_official_T1_seed42", "v": 1},
            {"_dir": "results/main/rf_official_T1_seed42__v2", "v": 2},
            {"_dir": "results/main/rf_official_T1_seed43", "v": 1},
            {"_dir": "results/ablation/rf_official_T1_seed42", "v": 1}]
    kept = sorted(r["_dir"] for r in latest_runs(runs))
    assert kept == ["results/ablation/rf_official_T1_seed42", "results/main/rf_official_T1_seed42__v2",
                    "results/main/rf_official_T1_seed43"]


def test_run_done_needs_a_finished_copy(tmp_path):
    from trace_iomt.experiment import run_done, run_name
    cfg = {"paths": {"results_dir": str(tmp_path)}}
    name = run_name("trace_moe", "loao", 16, "ARP_Spoofing", 42)
    assert name == "trace_moe_loao_T16_ARP_Spoofing_seed42"
    d = tmp_path / "zero_day" / name
    d.mkdir(parents=True)
    assert not run_done(cfg, "zero_day", name)                         # folder without metrics.json
    (d / "metrics.json").write_text('{"test": {}}')
    assert run_done(cfg, "zero_day", name)
    assert not run_done(cfg, "zero_day", name, required_key="ood")     # finished training, OOD block missing
    (tmp_path / "zero_day" / f"{name}__v2").mkdir()
    (tmp_path / "zero_day" / f"{name}__v2" / "metrics.json").write_text('{"ood": {}}')
    assert run_done(cfg, "zero_day", name, required_key="ood")         # the newest copy counts


def test_collect_reads_only_the_newest_copy_of_repeated_main_runs(tmp_path):
    """Repeated runs (<name>__vN) must not be counted twice in the H5/H9 inputs (abstain, routing)."""
    import json
    from trace_iomt.reporting import abstain_and_routing
    for name, nmi in [("trace_moe_official_T16_seed42", 0.1), ("trace_moe_official_T16_seed42__v2", 0.5),
                      ("trace_moe_official_T16_seed43", 0.7)]:
        d = tmp_path / "main" / name
        d.mkdir(parents=True)
        seed = int(name.split("seed")[1][:2])
        (d / "metrics.json").write_text(json.dumps({
            "model": "trace_moe", "regime": "official", "seed": seed,
            "abstain": {"aurc": nmi}, "routing": {"nmi_label6": nmi, "nmi_label19": nmi}}))
    abstain, routing = abstain_and_routing(tmp_path)
    assert sorted(routing["nmi_label6"]) == [0.5, 0.7]
    assert sorted(abstain["aurc"]) == [0.5, 0.7]
