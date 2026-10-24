import hashlib
import io
import json
import os
import stat
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("gdown")

from ntv_ai.data import download as dl


@pytest.fixture
def server():
    routes = {}
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            status, data, content_type = routes.get(self.path, (404, b"missing", "text/plain"))
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", routes, hits
    httpd.shutdown()
    thread.join()
    httpd.server_close()


def spec(url, data, path="arquivo.bin"):
    return {"url": url, "path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_download_reuses_checked_file_and_preserves_divergent_file(tmp_path, server):
    url, routes, hits = server
    routes["/file"] = (200, b"dataset", "application/octet-stream")
    item = spec(url + "/file", b"dataset", "base com espaços/dados.bin")
    assert not dl.fetch_file(tmp_path, item)["reused"]
    assert dl.fetch_file(tmp_path, item)["reused"]
    assert hits == ["/file"]
    target = tmp_path / item["path"]
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="preservado"):
        dl.fetch_file(tmp_path, item)
    assert target.read_bytes() == b"changed"
    assert hits == ["/file"]


@pytest.mark.parametrize("status,data,content_type", [
    (404, b"missing", "text/plain"),
    (200, b"<html>quota</html>", "text/html"),
    (200, b"<html>quota</html>", "application/octet-stream"),
    (200, b"wrong", "application/octet-stream"),
])
def test_bad_download_never_publishes_file(tmp_path, server, status, data, content_type):
    url, routes, _ = server
    routes["/file"] = (status, data, content_type)
    with pytest.raises(Exception):
        dl.fetch_file(tmp_path, spec(url + "/file", b"expected"), attempts=1)
    assert not list(tmp_path.iterdir())


def test_retry_after_network_failure(tmp_path, monkeypatch):
    attempts = []

    def transfer(url, target, **kwargs):
        attempts.append(url)
        target.write_bytes(b"partial" if len(attempts) == 1 else b"complete")
        if len(attempts) == 1:
            raise OSError("connection closed")

    monkeypatch.setattr(dl, "stream_download", transfer)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    dl.fetch_file(tmp_path, spec("https://example.org/file", b"complete"))
    assert len(attempts) == 2
    assert (tmp_path / "arquivo.bin").read_bytes() == b"complete"


def test_existing_file_without_receipt_is_compared_with_source(tmp_path, server):
    url, routes, _ = server
    routes["/file"] = (200, b"original", "text/plain")
    item = {"url": url + "/file", "path": "label.txt"}
    target = tmp_path / "label.txt"
    target.write_bytes(b"original")
    assert dl.fetch_file(tmp_path, item)["reused"]
    target.write_bytes(b"local")
    with pytest.raises(ValueError, match="preservado"):
        dl.fetch_file(tmp_path, item)
    assert target.read_bytes() == b"local"


@pytest.mark.parametrize("name", ["../file", "/file", r"C:\file", r"C:file", r"\\host\share\file",
                                  "NUL.txt", "dir/con", "file:stream", "trailing. ", "a\x00b"])
def test_rejects_unsafe_or_windows_invalid_names(name):
    with pytest.raises(ValueError):
        dl.relative_path(name)


def test_windows_relative_paths_and_case_collisions():
    assert dl.relative_path(r"MPID\glass_MPID\train\images\a.jpg") == "MPID/glass_MPID/train/images/a.jpg"
    with pytest.raises(ValueError, match="Windows"):
        dl.check_names(["Images/a.jpg", "images/b.jpg"])


@pytest.mark.parametrize("format", ["zip", "tar"])
def test_extract_layout_resume_and_preservation(tmp_path, format):
    archive = tmp_path / ("sample.zip" if format == "zip" else "sample.tar.bz2")
    payloads = {"dataset/images/a.jpg": b"image", "dataset/labels/a.txt": b"label"}
    if format == "zip":
        with zipfile.ZipFile(archive, "w") as container:
            for path, data in payloads.items():
                container.writestr(path, data)
    else:
        with tarfile.open(archive, "w:bz2") as container:
            for path, data in payloads.items():
                member = tarfile.TarInfo(path)
                member.size = len(data)
                container.addfile(member, io.BytesIO(data))
    root = tmp_path / "saída com espaços"
    assert dl.extract_archive(archive, root) == 2
    image = root / "dataset/images/a.jpg"
    modified = image.stat().st_mtime_ns
    (root / "dataset/labels/a.txt").unlink()
    assert dl.extract_archive(archive, root) == 2
    assert image.stat().st_mtime_ns == modified
    image.write_bytes(b"altered")
    with pytest.raises(ValueError, match="preservado"):
        dl.extract_archive(archive, root)
    assert image.read_bytes() == b"altered"


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate", "case"])
def test_rejects_bad_zip_before_extracting(tmp_path, kind):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as container:
        container.writestr("valid.txt", b"valid")
        if kind == "traversal":
            container.writestr("../escape", b"bad")
        elif kind == "symlink":
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            container.writestr(info, "../escape")
        elif kind == "duplicate":
            with pytest.warns(UserWarning):
                container.writestr("valid.txt", b"different")
        else:
            container.writestr("VALID.txt", b"different")
    with pytest.raises(ValueError):
        dl.extract_archive(archive, tmp_path / "raw")
    assert not list((tmp_path / "raw").rglob("*"))


def test_rejects_tar_links(tmp_path):
    archive = tmp_path / "links.tar"
    with tarfile.open(archive, "w") as container:
        item = tarfile.TarInfo("link")
        item.type = tarfile.SYMTYPE
        item.linkname = "../escape"
        container.addfile(item)
    with pytest.raises(ValueError):
        dl.extract_archive(archive, tmp_path / "raw")


def test_drive_inventory_cache_and_windows_paths(tmp_path, monkeypatch):
    import gdown

    def listing(**kwargs):
        assert kwargs["skip_download"]
        return [SimpleNamespace(id="abc_123", path=r"glass_MPID\data.yaml")]

    monkeypatch.setattr(gdown, "download_folder", listing)
    config = {"folder": "https://drive.google.com/drive/folders/abc", "destination": "raw/MPID"}
    files = dl.folder_files(tmp_path, config)
    assert files[0]["path"] == "raw/MPID/glass_MPID/data.yaml"
    monkeypatch.setattr(gdown, "download_folder", lambda **kwargs: pytest.fail("listed again"))
    assert dl.folder_files(tmp_path, config) == files
    with pytest.raises(ValueError, match="mudou"):
        dl.folder_files(tmp_path, {**config, "folder": "different"})


def test_mpid_failure_resume_and_metadata(tmp_path, server, monkeypatch):
    url, routes, hits = server
    routes["/good"] = (200, b"good", "text/plain")
    files = [{"path": "raw/MPID/good.txt", "url": url + "/good"},
             {"path": "raw/MPID/bad.txt", "url": url + "/bad"}]
    monkeypatch.setattr(dl, "folder_files", lambda *args: files)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    config = {"source": "https://example.org/mpid", "folder": "folder"}
    status = dl.download_dataset("mpid", config, tmp_path)
    assert status["status"] == "failed"
    assert status["files"] == 1
    assert status["errors"][0]["path"] == "raw/MPID/bad.txt"
    routes["/bad"] = (200, b"fixed connection", "text/plain")
    status = dl.download_dataset("mpid", config, tmp_path)
    assert status["status"] == "completed"
    assert status["reused"] == 1
    assert hits.count("/good") == 1
    metadata = tmp_path / "mpid/metadata"
    records = json.loads((metadata / "download-files.json").read_text())
    assert all(record["sha256"] for record in records.values())
    assert json.loads((metadata / "download-status.json").read_text())["status"] == "completed"


def test_legacy_mpid_hashes_do_not_require_personal_paths(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "acquisition.json").write_text(json.dumps({"artifacts": [
        {"path": "/old/machine/MPID/glass/file.jpg", "source_path": "glass/file.jpg",
         "bytes": 5, "sha256": "abc"}]}))
    assert dl.previous_files(tmp_path) == {"raw/MPID/glass/file.jpg": {"bytes": 5, "sha256": "abc"}}


@pytest.mark.parametrize("args,expected", [([], list(dl.DATASETS)),
    (["--datasets", "MPID", "STN-PLAD", "mpid"], ["mpid", "stn_plad"])])
def test_cli_selection_default_and_aliases(monkeypatch, args, expected, tmp_path):
    calls = []

    def run(name, config, output, **kwargs):
        calls.append(name)
        assert output == tmp_path
        return {"status": "completed", "reused": 0}

    monkeypatch.setattr(dl, "download_dataset", run)
    project = Path(__file__).resolve().parents[1]
    assert dl.main([*args, "--output", str(tmp_path)], project_root=project) == 0
    assert calls == expected


def test_cli_reports_failure_and_continues_other_datasets(monkeypatch, tmp_path):
    calls = []

    def run(name, *args, **kwargs):
        calls.append(name)
        return {"status": "failed", "errors": [{"error": "network"}]}

    monkeypatch.setattr(dl, "download_dataset", run)
    assert dl.main(["--datasets", "mpid", "ttpla", "--output", str(tmp_path)],
                   project_root=Path(__file__).resolve().parents[1]) == 1
    assert calls == ["mpid", "ttpla"]


def test_dataset_lock_prevents_parallel_writes(tmp_path):
    from filelock import FileLock, Timeout

    meta = tmp_path / "mpid/metadata"
    meta.mkdir(parents=True)
    with FileLock(str(meta / ".download.lock")):
        with pytest.raises(Timeout):
            dl.download_dataset("mpid", {"source": "example"}, tmp_path)


@pytest.mark.parametrize("portable", [False, True])
def test_ptl_names_preserve_coco_references(tmp_path, portable):
    if not portable and os.name == "nt":
        pytest.skip("Nomes originais com dois-pontos só podem ser extraídos no Linux")
    from PIL import Image
    from ntv_ai.data.adapters.readers import CocoReader

    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="JPEG")
    original_names = ["frame#00:15:05.jpg", "frame#00%3A15%3A05.jpg"]
    annotations = {"categories": [{"id": 0, "name": "insulator_ok"}],
                   "images": [{"id": i, "file_name": name, "width": 32, "height": 32}
                              for i, name in enumerate(original_names)],
                   "annotations": [{"id": i, "image_id": i, "category_id": 0, "bbox": [1, 2, 10, 12]}
                                   for i in range(2)]}
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as container:
        for name in original_names:
            container.writestr("imgs/train/" + name, image.getvalue())
        container.writestr("annotations.json", json.dumps(annotations))
    root = tmp_path / "ptl_ai"
    renamed = {}
    dl.extract_archive(archive, root / "raw/dataset", portable_names=portable, renamed=renamed)
    if portable:
        assert renamed["imgs/train/frame#00%3A15%3A05.jpg"] == "imgs/train/frame#00:15:05.jpg"
        dl.save_json(root / "metadata/download-paths.json", {"schema_version": 1, "paths": {
            "raw/dataset/" + local: "raw/dataset/" + original for local, original in renamed.items()}})
    else:
        assert not renamed
    reader = CocoReader({"id": "ptl_ai", "images": ["raw/dataset/imgs/train/*.jpg"],
                         "annotations": ["raw/dataset/annotations.json"]}, root)
    records = reader.read()
    assert len(records) == 2
    assert sum(len(record.annotations) for record in records) == 2
    assert not reader.issues
    for record in records:
        assert (root / record.path).read_bytes() == image.getvalue()
        assert record.annotations[0].geometry.coordinates == [1, 2, 11, 14]
    assert json.loads((root / "raw/dataset/annotations.json").read_text()) == annotations
