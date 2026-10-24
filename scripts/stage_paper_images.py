"""Find the frozen paper images by SHA-256 and copy them into portable source roots."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from paper import frozen,required
from ntv_ai.storage import file_hash,atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--study',choices=['all','main','external','mixed','annotation','vlm'],default='main')
    a=p.parse_args();output=a.output.resolve()
    for source in a.search:
        if output==source.resolve() or output.is_relative_to(source.resolve()):
            raise ValueError('Use an output directory outside the searched originals')
    wanted={};roots={}
    for key,_ in required(a.study):
        _,records=frozen(key)
        for r in records:
            if r.metadata.get('transform'):continue
            roots[r.dataset_id]=str(output/r.dataset_id)
            target=(output/r.dataset_id/r.path).resolve()
            if not target.is_relative_to(output):raise ValueError('Unsafe relative path')
            wanted.setdefault(r.sha256,set()).add(target)
    missing=set(wanted)
    for digest,targets in wanted.items():
        if all(p.is_file() and file_hash(p)==digest for p in targets):missing.discard(digest)
    for folder in a.search:
        for path in folder.rglob('*'):
            if not missing:break
            if not path.is_file() or path.suffix.lower() not in ('.jpg','.jpeg','.png','.tif','.tiff'):continue
            digest=file_hash(path)
            if digest not in missing:continue
            for target in wanted[digest]:
                if target.exists():
                    if file_hash(target)!=digest:raise ValueError(f'Refusing to overwrite {target}')
                else:
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            missing.remove(digest)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'roots.json',roots,replace=True)
    atomic_json(output/'staging.json',{'required_unique_images':len(wanted),'missing_sha256':sorted(missing)},replace=True)
    print(json.dumps({'required_unique_images':len(wanted),'missing':len(missing),'roots':str(output/'roots.json')},indent=2))
    if missing:raise SystemExit('Some exact paper images are missing; see staging.json. No substitute images were selected.')

if __name__=='__main__':main()
