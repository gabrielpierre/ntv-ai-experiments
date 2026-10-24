"""Deterministic temporal/visual grouping with optional embargo images."""
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

from ntv_ai.storage import digest


def timestamp(value):
    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")


def fingerprints(image):
    import numpy as np
    from scipy.fft import dctn
    from PIL import Image
    gray = image.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(small, type=2, norm="ortho")[:8, :8].flatten()
    ph = low > np.median(low[1:])
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    dh = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    def integer(bits):
        return sum(int(v) << i for i, v in enumerate(bits))
    return {"phash": integer(ph), "dhash": integer(dh)}


def build_groups(records, anchors, options):
    groups, pairs, edges = defaultdict(list), [], set()
    seconds = options.get("block_seconds", 60)
    if seconds <= 0:
        raise ValueError("block_seconds must be positive")
    for r in records:
        meta = r.metadata
        capture = meta["capture_id"]
        elapsed = (timestamp(meta["datetime"]) - timestamp(anchors[capture])).total_seconds()
        block = int(elapsed // seconds)
        r.group_id = f"{capture}:block-{block:06d}"
        meta.update(grouping_reviewed=True, grouping_basis="temporal_visual_algorithm_v1")
        groups[r.group_id].append(r)
    for a, b in combinations(sorted(records, key=lambda r: r.image_id), 2):
        am, bm = a.metadata, b.metadata
        dt = abs((timestamp(am["datetime"]) - timestamp(bm["datetime"])).total_seconds())
        close = am["capture_id"] == bm["capture_id"] and dt < options.get("guard_seconds", 15)
        ph = (am["phash"] ^ bm["phash"]).bit_count()
        dh = (am["dhash"] ^ bm["dhash"]).bit_count()
        exact = a.sha256 == b.sha256 or bool(a.pixel_sha256 and a.pixel_sha256 == b.pixel_sha256)
        visual = (ph <= options.get("phash_distance", 6) and dh <= options.get("dhash_distance", 6)) or exact
        if close or visual:
            pairs.append({"a": a.image_id, "b": b.image_id, "temporal": close,
                          "visual": visual, "phash_distance": ph, "dhash_distance": dh})
            if a.group_id != b.group_id:
                edges.add(tuple(sorted((a.group_id, b.group_id))))
    return dict(sorted(groups.items())), sorted(edges), pairs


def grouped_split(records, anchors, classes, options, previous=None):
    """MILP: each block train/val/buffer; related blocks cannot cross train/val."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    groups, edges, pairs = build_groups(records, anchors, options)
    keys = list(groups)
    indices = {key: i for i, key in enumerate(keys)}
    captures = sorted({r.metadata["capture_id"] for r in records})
    # Three binary assignments per block, followed by absolute-deviation variables.
    count = 3 * len(keys)
    size = count + 1 + len(captures)
    c = np.zeros(size)
    upper = np.ones(size); upper[count:] = np.inf
    integrality = np.zeros(size); integrality[:count] = 1
    for i, key in enumerate(keys):
        c[3*i+2] = 100 * len(groups[key])
        # Stable tie break, independent of filesystem/input order.
        c[3*i+1] = int(digest([options.get("seed", 43), key])[:8], 16) / 2**32 * 0.00001
    c[count] = 20
    c[count+1:] = 1
    constraints = []
    def add(row, lower=-np.inf, high=np.inf):
        constraints.append((row, lower, high))
    for i in range(len(keys)):
        add({3*i+s: 1 for s in range(3)}, 1, 1)
    for a, b in edges:
        ai, bi = indices[a], indices[b]
        add({3*ai: 1, 3*bi+1: 1}, high=1)
        add({3*bi: 1, 3*ai+1: 1}, high=1)
    previous = previous or {}
    previous_groups = previous.get("group_assignment", {})
    previous_images = previous.get("assignment", previous)
    for key, members in groups.items():
        fixed = {previous_images[r.sha256] for r in members if r.sha256 in previous_images}
        if key in previous_groups:
            fixed.add(previous_groups[key])
        if len(fixed) > 1:
            raise ValueError("New grouping conflicts with frozen partition")
        if fixed:
            value = fixed.pop()
            if value in ("train", "val", "buffer"):
                add({3*indices[key]+("train", "val", "buffer").index(value): 1}, 1, 1)
    ratio = options.get("validation_fraction", 0.2)
    for ci, capture in enumerate([None, *captures]):
        weights = {3*i+1: sum(capture is None or r.metadata["capture_id"] == capture
                              for r in groups[k]) for i, k in enumerate(keys)}
        total = sum(weights.values())
        dev = count + ci
        add({**weights, dev: -1}, high=ratio * total)
        add({**weights, dev: 1}, lower=ratio * total)
        for split in (0, 1):
            add({j-1+split: n for j, n in weights.items()}, lower=1)
    hard_end = len(constraints)
    for label in classes:
        for split in (0, 1):
            objects, positive = {}, {}
            for i, key in enumerate(keys):
                n = sum(a.label == label for r in groups[key] for a in r.annotations)
                objects[3*i+split] = n
                positive[3*i+split] = int(n > 0)
            add(objects, lower=options.get("min_objects", 10))
            add(positive, lower=options.get("min_positive_groups", 2))
    def solve(rows):
        matrix = lil_matrix((len(rows), size))
        for i, (row, _, _) in enumerate(rows):
            for j, v in row.items():
                matrix[i, j] = v
        return milp(c, integrality=integrality, bounds=Bounds(np.zeros(size), upper),
                    constraints=LinearConstraint(matrix.tocsr(), [r[1] for r in rows], [r[2] for r in rows]),
                    options={"time_limit": 60, "mip_rel_gap": 0})
    solution = solve(constraints)
    warnings = []
    if solution.x is None:
        # Never relax leakage constraints. Keep at least one group/class when feasible.
        warnings.append("Requested class support infeasible; preserve grouping and report deficits")
        reduced = constraints[:hard_end]
        for label in classes:
            for split in (0, 1):
                row = {3*i+split: int(any(a.label == label for r in groups[k] for a in r.annotations))
                       for i, k in enumerate(keys)}
                if sum(row.values()) >= 2:
                    reduced.append((row, 1, np.inf))
        solution = solve(reduced)
        if solution.x is None:
            raise ValueError("No safe train/val partition; review support/groups, never split related frames")
    for i, key in enumerate(keys):
        split = ("train", "val", "buffer")[int(np.argmax(solution.x[3*i:3*i+3]))]
        for r in groups[key]:
            r.split = split if split != "buffer" else None
            r.metadata["excluded"] = split == "buffer"
            r.metadata["exclusion_reason"] = "temporal_visual_buffer" if split == "buffer" else None
    check_partition(records, edges)
    return {"options": options, "groups": {k: [r.image_id for r in v] for k, v in groups.items()},
            "group_assignment": {k: v[0].split or "buffer" for k, v in groups.items()},
            "edges": edges, "pairs": pairs, "support": support(records, classes), "warnings": warnings,
            "assignment": {r.sha256: r.split or "buffer" for r in records}}


def support(records, classes):
    def summary(rows):
        return {"images": len(rows), "boxes": sum(len(r.annotations) for r in rows),
                "captures": dict(Counter(r.metadata["capture_id"] for r in rows)),
                "classes": {c: {"objects": sum(a.label == c for r in rows for a in r.annotations),
                                  "images": sum(any(a.label == c for a in r.annotations) for r in rows),
                                  "groups": len({r.group_id for r in rows if any(a.label == c for a in r.annotations)})}
                            for c in classes}}
    return {"all": summary(records), **{s: summary([r for r in records if (r.split or "buffer") == s])
                                         for s in ("train", "val", "test", "buffer")}}


def check_partition(records, edges):
    occupied = defaultdict(set)
    identities = defaultdict(set)
    for r in records:
        if r.split:
            occupied[r.group_id].add(r.split)
            for kind, value in (("sha", r.sha256), ("pixels", r.pixel_sha256)):
                if value:
                    identities[(kind, value)].add(r.split)
    if any(len(v) > 1 for v in [*occupied.values(), *identities.values()]):
        raise ValueError("Identity/group leakage")
    for a, b in edges:
        if len(occupied[a] | occupied[b]) > 1:
            raise ValueError("Temporal/visual leakage")
