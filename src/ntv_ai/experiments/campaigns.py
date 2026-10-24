import copy
import statistics
from pathlib import Path

from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.recipes import compatibility, positive_int, resolve_recipe
from ntv_ai.experiments.runs import Run, now
from ntv_ai.storage import atomic_json, digest, load_json


def summarize(path):
    path = Path(path)
    state = load_json(path / "campaign.json")
    groups = {}
    for item in state["experiments"]:
        if item["state"] != "completed":
            continue
        result = item["result"]
        metrics = result.get("metrics")
        if metrics is None:
            selection = result.get("best")
            metrics = {"mAP50_95": selection["mAP50_95"]} if selection else None
        if metrics is None or metrics.get("mAP50_95") is None:
            continue
        runtime = load_json(Path(result["run_path"]) / "runtime.json")
        actual_environment = {key: runtime.get(key) for key in ("dependencies", "gpu", "torch", "cuda", "python", "code_digest")}
        key = digest({"recipe": result["compatibility"], "environment": actual_environment})
        group = groups.setdefault(key, {"compatibility": result["compatibility"], "backend": item["backend"],
                                      "values": [], "seeds": [], "runs": []})
        if item["seed"] in group["seeds"]:
            raise ValueError("Seed duplicada no mesmo grupo de agregação")
        group["values"].append(metrics["mAP50_95"])
        group["seeds"].append(item["seed"])
        group["runs"].append(result["run_id"])
    for group in groups.values():
        group.update(n=len(group["values"]), mean=statistics.mean(group["values"]),
                     sample_std=statistics.stdev(group["values"]) if len(group["values"]) > 1 else None)
    counts = {name: sum(x["state"] == name for x in state["experiments"])
              for name in ("pending", "running", "completed", "failed", "interrupted")}
    report = {"purpose": state["purpose"], "counts": counts, "complete": not any(counts[k] for k in ("pending", "running", "interrupted")),
              "all_passed": counts["completed"] == len(state["experiments"]), "groups": groups,
              "failures": [x for x in state["experiments"] if x["state"] in ("failed", "interrupted")]}
    atomic_json(path / "summary.json", report, replace=True)
    return report


def campaign(config_path, storage_root=None, runs_root=None, resume=None):
    from ntv_ai.pipelines.experiments import launch
    if resume:
        folder = Path(resume).resolve()
        state = load_json(folder / "campaign.json")
        if any(item["state"] == "running" for item in state["experiments"]):
            # Require the caller to inspect interrupted workers instead of launching duplicates.
            raise ValueError("Campanha contém worker marcado running; revise o estado antes de continuar")
        run_context = None
    else:
        config_path = Path(config_path).resolve()
        config = read_config(config_path)
        seeds = config["seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("Seeds da campanha devem ser únicas e não vazias")
        if config["operation"] not in ("smoke", "train"):
            raise ValueError("Operação de campanha inválida")
        for seed in seeds:
            positive_int(seed, "seed", 0)
        experiments = []
        for recipe in config["recipes"]:
            resolved = resolve_recipe(config_path.parent / recipe, storage_root)
            if resolved["purpose"] != config["purpose"]:
                raise ValueError("Finalidades incompatíveis na campanha")
            for seed in seeds:
                experiment = copy.deepcopy(resolved)
                experiment["training"]["seed"] = seed
                experiments.append({"recipe": recipe, "seed": seed, "config": experiment,
                                    "backend": experiment["model"]["backend"], "state": "pending"})
        if not experiments:
            raise ValueError("Campanha vazia")
        root = Path(runs_root or Path(experiments[0]["config"]["runtime"]["storage_root"]) / "runs")
        run_context = Run(root, "campaign", config)
        folder = run_context.path
        state = {"purpose": config["purpose"], "operation": config["operation"], "experiments": experiments}
        atomic_json(folder / "campaign.json", state)
    try:
        for item in state["experiments"]:
            if item["state"] not in ("pending", "interrupted"):
                continue
            item.update(state="running", started_at=now())
            atomic_json(folder / "campaign.json", state, replace=True)
            try:
                item["result"] = launch(state["operation"], item["config"], runs_root=runs_root)
                item["state"] = "completed"
            except Exception as exc:
                item.update(state="failed", error=str(exc))
            except KeyboardInterrupt:
                item.update(state="interrupted", error="Interrompido pelo usuário")
                raise
            finally:
                item["finished_at"] = now()
                atomic_json(folder / "campaign.json", state, replace=True)
        report = summarize(folder)
        if run_context:
            run_context.__exit__(None, None, None)
        return {"campaign": str(folder), **report}
    except BaseException as exc:
        if run_context:
            run_context.__exit__(type(exc), exc, exc.__traceback__)
        raise
