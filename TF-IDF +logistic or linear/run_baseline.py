"""Train a policy-CV-selected TF-IDF logistic baseline under the original evaluation contract."""
import argparse,json,sys,time,warnings,csv
from pathlib import Path
import numpy as np
import sklearn,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parent/'shared'))
import bert_opp115_classification as backend
from opp115_hierarchy import HIGH_LEVEL_CATEGORIES as C,ATTRIBUTE_HEADS as H,derive_hierarchy_schema,fixed_policy_split
warnings.simplefilter('error',ConvergenceWarning)
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
train_ids,eval_ids=fixed_policy_split(args.data_dir/'annotations');schema=derive_hierarchy_schema(args.data_dir/'annotations',train_ids)
all_examples=backend.load_segment_examples(args.data_dir,schema);train,test,ti,ei=backend.split_segment_examples(all_examples)
assert list(ti)==list(train_ids) and not set(ti)&set(ei)
def arrays(examples):return np.array([e.category_labels for e in examples],bool),np.array([e.attribute_labels for e in examples],bool),np.array([e.attribute_mask for e in examples],bool)
cy,ay,mask=arrays(train);ety,eta,etmask=arrays(test);texts=[e.text for e in train]
def vectorizer():
    return FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,strip_accents='unicode')),('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=60000,sublinear_tf=True))])
def fit_heads(x,cy,ay,mask,c):
    estimators=[]
    targets=np.concatenate([cy,ay],1)
    active=[np.ones(len(cy),bool) for _ in C]
    for j,h in enumerate(H):
        a,b=schema.attribute_slices[h];active.extend([mask[:,j]]*(b-a))
    assert len(active)==targets.shape[1]
    for k,sel in enumerate(active):
        y=targets[sel,k].astype(int)
        if len(np.unique(y))<2:estimators.append(float(y[0]) if len(y) else 0.0);continue
        model=LogisticRegression(C=c,solver='liblinear',class_weight='balanced',max_iter=3000,random_state=1729)
        model.fit(x[sel],y);estimators.append(model)
    return estimators
def predict(estimators,x):return np.column_stack([np.full(x.shape[0],m) if isinstance(m,float) else m.predict_proba(x)[:,1] for m in estimators])
def metrics(prob,cy,ay,mask):
    cp=prob[:,:len(C)]>=.5;ap=prob[:,len(C):]>=.5;gated=backend.gate_attribute_predictions(ap,cp,schema)
    cm=backend.compute_multilabel_classification_metrics(cy,cp,list(C));fm=backend.compute_attribute_metrics(ay,gated,mask,schema);oracle=backend.compute_attribute_metrics(ay,ap,mask,schema)
    return dict(category=cm,fine_grained=fm,independent_attribute_heads=oracle)
folds=list(GroupKFold(n_splits=3).split(texts,groups=[e.policy_id for e in train]));cv=[];started=time.time()
for fold,(fit,val) in enumerate(folds,1):
    v=vectorizer();x=v.fit_transform([texts[i] for i in fit]);z=v.transform([texts[i] for i in val]);assert not {train[i].policy_id for i in fit}&{train[i].policy_id for i in val}
    for c in [1.0,4.0,16.0]:
        heads=fit_heads(x,cy[fit],ay[fit],mask[fit],c);r=metrics(predict(heads,z),cy[val],ay[val],mask[val]);cat=r['category']['macro_f1'];fine=r['fine_grained']['fine_grained_macro_f1'];cv.append(dict(fold=fold,C=c,category_macro=cat,fine_macro=fine,selection_score=(cat+fine)/2,train_policies=len({train[i].policy_id for i in fit}),validation_policies=len({train[i].policy_id for i in val})))
        (out/'cv_progress.json').write_text(json.dumps(cv,indent=2));print('CV',cv[-1],flush=True)
scores={c:float(np.mean([r['selection_score'] for r in cv if r['C']==c])) for c in [1.0,4.0,16.0]};chosen=max(scores,key=lambda c:(scores[c],-c));v=vectorizer();x=v.fit_transform(texts);z=v.transform([e.text for e in test]);heads=fit_heads(x,cy,ay,mask,chosen);prob=predict(heads,z);result=metrics(prob,ety,eta,etmask)
summary=dict(model='TF-IDF + logistic regression',selected_C=chosen,cv_scores=scores,cv=cv,selection_metric='Mean of category Macro-F1 and gated fine-grained Macro-F1 across 3 training-policy folds',threshold=.5,seed=1729,train_policy_ids=list(ti),eval_policy_ids=list(ei),train_segments=len(train),eval_segments=len(test),feature_count=x.shape[1],features='Word 1–2 grams plus char_wb 3–5 grams; each branch <=60000 features; min_df=2; sublinear TF; per-branch L2 normalization',class_weight='balanced within each binary training problem',attribute_training='Only gold-applicable training segments per attribute head; constant predictor if only one class',schema_scope='Fixed from outer training policies; TF-IDF vocabularies and IDF refit inside every CV fold',text_scope='Full segment text, no transformer token truncation',sklearn_version=sklearn.__version__,elapsed_seconds=time.time()-started,results=result)
(out/'results.json').write_text(json.dumps(summary,indent=2));joblib.dump(dict(vectorizer=v,estimators=heads,schema=schema.to_dict() if hasattr(schema,'to_dict') else None),out/'baseline.joblib')
with (out/'prediction_rows.jsonl').open('w',encoding='utf-8') as f:
    for e,p in zip(test,prob):f.write(json.dumps(dict(policy_id=e.policy_id,segment_id=e.segment_id,text=e.text,gold_labels=[c for c,y in zip(C,e.category_labels) if y],probabilities=dict(zip(C,p[:len(C)].tolist())),attribute_labels=list(e.attribute_labels),attribute_mask=list(e.attribute_mask),attribute_probabilities=p[len(C):].tolist()),ensure_ascii=False)+'\n')
print('FINAL',chosen,result['category']['macro_f1'],result['fine_grained']['fine_grained_macro_f1'],flush=True)
