import csv
import io
import json
from pathlib import Path

from ntv_ai.data.audit import reconcile_counts
from ntv_ai.data.eligibility import build_report
from ntv_ai.experiments.config import read_config, reference
from ntv_ai.experiments.runs import Run
from ntv_ai.storage import atomic_json, digest, file_hash, inside, json_bytes, load_json, read_snapshot, safe_output


ACTIONS = {
    "incompatible_annotation_scope": "Revisar a cobertura de cada fonte e classe; declarar supervisão exaustiva somente com evidência.",
    "mapping_pending": "Decidir a equivalência dos rótulos pendentes com base na taxonomia e nas anotações originais.",
    "group_review_pending": "Confirmar grupos de duplicatas e capturas correlacionadas antes de definir os splits.",
    "group_crosses_splits": "Revisar os grupos que atravessam partições, incluindo relações entre fontes.",
    "split_missing": "Definir partições explícitas depois de revisar os grupos.",
    "bbox_conversion_not_authorized": "Decidir o uso das geometrias não BBox; polígonos e linhas permanecem preservados.",
    "coordinate_convention_unverified": "Confirmar a convenção de coordenadas da fonte antes de autorizar uma conversão.",
    "bbox_outside_export_bounds": "Inspecionar as caixas fora da imagem e registrar correções explícitas na curadoria.",
    "invalid_topology": "Inspecionar geometrias inválidas e registrar a decisão sem alterar os originais.",
    "unsupported_ignore_region": "Revisar imagens com regiões ignoradas; a supervisão atual não suporta essas regiões.",
    "image_integrity_mismatch": "Recuperar ou reimportar imagens ausentes ou com hash diferente do snapshot.",
    "annotation_integrity_mismatch": "Recuperar ou reimportar os arquivos de anotação ou metadados que divergiram.",
    "unresolved_source_issue": "Resolver as ocorrências de leitura da fonte; problemas sem alcance conhecido bloqueiam conclusões globais.",
}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown_report(report):
    totals = report["totals"]
    view = report["prepared_view"]
    lines = ["# Elegibilidade dos dados", "",
             f"Versão dos dados: `{report['dataset_version']}`.", "",
             f"{totals['images']} imagens originais e {totals['annotations']} anotações. "
             f"{totals['excluded_images']} imagens excluídas pela curadoria; "
             f"{totals['tiles_not_counted']} tiles fora das contagens.", "",
             f"View `{cell(view['id'])}`: **{'atende aos critérios registrados' if view['eligible_now'] else 'bloqueada'}**. "
             f"{view['blocked_images']} de {view['images_analyzed']} imagens têm bloqueios identificados.", "",
             "O comando apenas diagnostica os registros; não altera decisões nem libera exports.", ""]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines += ["", "## Fontes", "", "| Fonte | Imagens | Anotações | Imagens excluídas |",
              "| --- | ---: | ---: | ---: |"]
    for name, row in report["datasets"].items():
        lines.append(f"| {cell(name)} | {row['images']} | {row['annotations']} | {row['excluded_images']} |")
    if report.get("reconciliation"):
        matches = report["reconciliation"]["matches"]
        lines += ["", "Reconciliação com o inventário de referência: " + ("contagens coincidem." if matches else "há diferenças; consultar o JSON.")]
    lines += ["", "## Fonte e classe", "",
              "As contagens abaixo consideram imagens originais não excluídas. Uma imagem pode conter várias classes. "
              "Elegíveis incluem positivos e negativos com cobertura exaustiva explícita. "
              "Uma caixa válida, isoladamente, não torna a imagem elegível.", "",
              "| Fonte | Classe | Imagens com classe | Anotações mapeadas | BBoxes válidas | Elegíveis (+ / −) | Cobertura exaustiva / parcial / desconhecida |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["by_class"]:
        scope = row["coverage"]
        lines.append(f"| {cell(row['dataset_id'])} | {cell(row['class_id'])} ({row['mapping_level']}) | "
                     f"{row['images_with_class']} | {row['annotations_mapped']} | {row['valid_bbox_annotations']} | "
                     f"{row['eligible_images']} ({row['eligible_positive_images']} / {row['eligible_negative_images']}) | "
                     f"{scope['exhaustive']} / {scope['partial']} / {scope['unknown']} |")
    lines += ["", "## Bloqueios e geometrias", "",
              "As contagens de motivos podem se sobrepor; cada imagem conta uma vez por motivo e classe.", ""]
    for dataset in report["datasets"]:
        lines += [f"### {cell(dataset)}", ""]
        for row in report["by_class"]:
            if row["dataset_id"] != dataset or not row["annotations_mapped"]:
                continue
            geometry = ", ".join(f"{cell(k)}: {v}" for k, v in row["geometries"].items())
            reasons = ", ".join(f"`{cell(k)}`: {v}" for k, v in row["blocking_images_by_code"].items()) or "nenhum"
            lines.append(f"- **{cell(row['class_id'])}** — geometrias: {geometry}; imagens por bloqueio: {reasons}.")
        lines.append("")
    lines += ["## Rótulos pendentes ou excluídos", "",
              "| Fonte | Rótulo original | Estado | Anotações | Imagens | Motivo |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for row in report["original_labels"]:
        if row["mapping_status"] == "mapped" and not row["excluded_by_curation"] and not row["excluded_image"]:
            continue
        state = row["mapping_status"]
        if row["excluded_by_curation"]:
            state += "; anotação excluída"
        if row["excluded_image"]:
            state += "; imagem excluída"
        lines.append(f"| {cell(row['dataset_id'])} | {cell(row['original_label'])} | {cell(state)} | "
                     f"{row['annotations']} | {row['images']} | {cell('; '.join(row['reasons']))} |")
    gaps = ", ".join(f"`{cell(c)}`" for c in report["classes_without_mapped_occurrences"]) or "nenhuma"
    lines += ["", f"Classes sem ocorrências mapeadas nas imagens não excluídas: {gaps}.", "",
              "## Próximas decisões", ""]
    codes = {code for row in report["by_class"] for code in row["blocking_images_by_code"]}
    codes.update(report["global_blockers"])
    for code, action in ACTIONS.items():
        if code in codes:
            lines.append(f"- {action}")
    if report["reference_dataset"] and not report["reference_dataset"]["images_available"]:
        lines.append("- Disponibilizar as imagens e anotações da base de referência para validar a compatibilidade com o domínio.")
    lines += ["", "Detalhes: [JSON](eligibility.json), [tabela CSV](eligibility_by_class.csv) e "
              "[ocorrências por registro](eligibility_issues.jsonl).", ""]
    return "\n".join(lines)


def write_reports(folder, report, findings):
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(report["by_class"][0]) if report["by_class"] else ["dataset_id", "class_id"],
                            lineterminator="\n")
    writer.writeheader()
    for row in report["by_class"]:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    payloads = {
        "eligibility.md": markdown_report(report),
        "eligibility_by_class.csv": table.getvalue(),
        "eligibility_issues.jsonl": "".join(json_bytes(item).decode() for item in findings),
    }
    for name, content in payloads.items():
        with (folder / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    atomic_json(folder / "eligibility.json", report)


def run_eligibility(config_path: Path, manifest_path: Path, runs_root: Path | None = None):
    config = read_config(config_path)
    config = {**config, "manifest": str(manifest_path.resolve())}
    for key in ("taxonomy", "mapping", "baseline"):
        if config.get(key):
            config[key] = str(reference(config_path, config[key]))
    manifest_path = manifest_path.resolve()
    protected = [manifest_path.parent]
    try:
        header = load_json(manifest_path)
        protected.extend(Path(s["root"]) for s in header.get("sources", {}).values())
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    output = safe_output(runs_root or reference(config_path, config.get("runs", "../runs")), protected)
    config["runs"] = str(output)
    with Run(output, "eligibility", config) as run:
        manifest, records = read_snapshot(manifest_path)
        if manifest["lineage"].get("operation") != "prepare" or not manifest.get("view"):
            raise ValueError("Elegibilidade requer um snapshot produzido por prepare")
        taxonomy, mapping = (read_config(Path(config[key])) for key in ("taxonomy", "mapping"))
        for key, value in (("taxonomy", taxonomy), ("mapping", mapping)):
            if digest(value) != manifest["lineage"].get(key + "_sha256"):
                raise ValueError(f"{key} diverge da preparação; execute uma nova preparação")
        issues = load_json(inside(manifest_path.parent, manifest["issues"]))
        report, findings = build_report(manifest, records, taxonomy, issues, config.get("reference_dataset"))
        report["inputs"] = {
            "manifest": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
            "taxonomy_sha256": digest(taxonomy), "mapping_sha256": digest(mapping),
            "records_sha256": manifest["records_sha256"], "issues_sha256": manifest["issues_sha256"],
        }
        if config.get("baseline"):
            report["reconciliation"] = reconcile_counts(report, Path(config["baseline"]))
        write_reports(run.path, report, findings)
        run.status["artifacts"].update({name: str(run.path / name) for name in (
            "eligibility.md", "eligibility.json", "eligibility_by_class.csv", "eligibility_issues.jsonl")})
        return {"run_id": run.id, "report": str(run.path / "eligibility.md"),
                "totals": report["totals"], "eligible": report["prepared_view"]["eligible_now"]}
