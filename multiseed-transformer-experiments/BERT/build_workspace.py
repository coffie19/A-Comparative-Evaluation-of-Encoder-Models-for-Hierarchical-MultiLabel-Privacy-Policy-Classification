"""Rebuild this model's portable multi-seed archive and integrity manifest."""
from pathlib import Path
import hashlib
import json
import zipfile


def build(root):
    root=Path(root)
    model=json.loads((root / "model_config.json").read_text())["model"]
    files=[]
    for p in sorted(root.rglob("*")):
        relative=p.relative_to(root)
        if not p.is_file() or any(part in {"__pycache__","runs","prepared",".git"} for part in relative.parts): continue
        if p.suffix in {".zip",".pyc"} or p.name.endswith((".sha256",".manifest.json")) or p.name=="workspace_manifest.json": continue
        files.append(p)
    records=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    manifest=dict(format_version=1,model=model,study="matched-multi-seed",files=records)
    encoded=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    (root / "workspace_manifest.json").write_bytes(encoded)
    archive=root / f"{model}_multiseed_workspace.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info=zipfile.ZipInfo(p.relative_to(root).as_posix(),date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644 << 16
            z.writestr(info,p.read_bytes())
        z.writestr("workspace_manifest.json",encoded)
    archive.with_suffix(".manifest.json").write_bytes(encoded)
    archive.with_suffix(".zip.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+"  "+archive.name+"\n")
    return archive


if __name__ == "__main__": print(build(Path(__file__).resolve().parent))
