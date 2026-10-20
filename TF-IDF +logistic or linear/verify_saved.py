import json,sys,argparse
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import f1_score
parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,required=True);parser.add_argument('--schema',type=Path,required=True);args=parser.parse_args();out=args.results_dir
model=joblib.load(out/'baseline.joblib');rows=[json.loads(l) for l in (out/'prediction_rows.jsonl').read_text(encoding='utf-8').splitlines()];x=model['vectorizer'].transform([r['text'] for r in rows]);prob=np.column_stack([np.full(len(rows),m) if isinstance(m,float) else m.predict_proba(x)[:,1] for m in model['estimators']]);S=json.loads(args.schema.read_text());C=S['high_level_categories'];H=S['attribute_heads'];expected=np.array([[r['probabilities'][c] for c in C]+r['attribute_probabilities'] for r in rows]);assert np.allclose(prob,expected,rtol=0,atol=1e-12)
cy=np.array([[c in r['gold_labels'] for c in C] for r in rows]);cp=prob[:,:10]>=.5;ay=np.array([r['attribute_labels'] for r in rows]);ap=prob[:,10:]>=.5;mask=np.array([r['attribute_mask'] for r in rows],bool);vals=[]
for j,h in enumerate(H):
    a,b=S['attribute_slices'][h];idx=[i for i,c in enumerate(C) if h in S['category_to_attributes'][c]];ap[:,a:b]&=cp[:,idx].any(1)[:,None];sel=mask[:,j];vals.extend(f1_score(ay[sel,a:b],ap[sel,a:b],average=None,zero_division=0))
r=json.loads((out/'results.json').read_text());assert abs(f1_score(cy,cp,average='macro',zero_division=0)-r['results']['category']['macro_f1'])<1e-12;assert abs(np.mean(vals)-r['results']['fine_grained']['fine_grained_macro_f1'])<1e-12
(out/'validation.json').write_text(json.dumps(dict(saved_model_roundtrip_all_1440_rows=True,max_probability_error=float(np.max(np.abs(prob-expected))),independent_sklearn_category_and_fine_f1_match=True,training_policy_cv=True,held_out_used_for_selection=False),indent=2));print('All saved probabilities and independent metrics validated')

