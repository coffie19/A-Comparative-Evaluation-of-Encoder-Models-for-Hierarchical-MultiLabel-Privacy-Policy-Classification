import argparse,csv,json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--baseline-dir',type=Path,required=True);parser.add_argument('--transformer-predictions',type=Path,required=True);parser.add_argument('--exclusions',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--schema',type=Path,required=True);args=parser.parse_args()
S=json.loads(args.schema.read_text());cats=S['high_level_categories'];heads=S['attribute_heads'];out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
excluded=set(json.loads(args.exclusions.read_text())['normalized_plus_near'])
def f1(y,p):
    tp=(y&p).sum(0);fp=(~y&p).sum(0);fn=(y&~p).sum(0);den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den>0)
def write(name,rr):
    with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
totals=[];category=[];attribute=[];baseline_keys=None
for model in ['TF-IDF + logistic regression','BERT','RoBERTa','LegalBERT','PrivBERT']:
    p=args.baseline_dir/'prediction_rows.jsonl' if model.startswith('TF-IDF') else args.transformer_predictions/model/'prediction_rows.jsonl'
    rr=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()];keys=[r['policy_id']+':'+str(r['segment_id']) for r in rr]
    if baseline_keys is None:baseline_keys=keys;baseline_truth=[(r['gold_labels'],r['attribute_labels'],r['attribute_mask']) for r in rr]
    else:assert keys==baseline_keys and [(r['gold_labels'],r['attribute_labels'],r['attribute_mask']) for r in rr]==baseline_truth
    cy=np.array([[c in r['gold_labels'] for c in cats] for r in rr]);cp=np.array([[r['probabilities'][c]>=.5 for c in cats] for r in rr]);ay=np.array([r['attribute_labels'] for r in rr],bool);raw=np.array([r['attribute_probabilities'] for r in rr])>=.5;mask=np.array([r['attribute_mask'] for r in rr],bool);ap=raw.copy()
    for h in heads:
        a,b=S['attribute_slices'][h];idx=[i for i,c in enumerate(cats) if h in S['category_to_attributes'][c]];ap[:,a:b]&=cp[:,idx].any(1)[:,None]
    for condition in ['original','duplicate_filtered']:
        keep=np.ones(len(rr),bool) if condition=='original' else np.array([k not in excluded for k in keys]);cf=f1(cy[keep],cp[keep]);binary=(cf+f1(~cy[keep],~cp[keep]))/2;af=[];of=[]
        for i,c in enumerate(cats):category.append(dict(model=model,condition=condition,category=c,positive_f1=float(cf[i]),presence_absence_macro_f1=float(binary[i]),support=int(cy[keep,i].sum())))
        for j,h in enumerate(heads):
            a,b=S['attribute_slices'][h];sel=keep&mask[:,j];vv=f1(ay[sel,a:b],ap[sel,a:b]);af.extend(vv);ov=f1(ay[sel,a:b],raw[sel,a:b]);of.extend(ov);attribute.append(dict(model=model,condition=condition,head=h,macro_f1=float(vv.mean()),oracle_macro_f1=float(ov.mean()),n=int(sel.sum())))
        totals.append(dict(model=model,condition=condition,n=int(keep.sum()),category_macro_f1=float(cf.mean()),fine_macro_f1=float(np.mean(af)),oracle_fine_macro_f1=float(np.mean(of)),paper_style_nine_category_macro=float(binary[:9].mean())))
write('matched_comparison',totals);write('category_comparison',category);write('attribute_comparison',attribute)
query=[('First Party Collection/Use',.80,1267),('Third Party Sharing/Collection',.81,963),('User Choice/Control',.75,455),('Data Security',.87,202),('International and Specific Audiences',.95,156),('Access, Edit and Deletion',.82,134),('Policy Change',.92,120),('Data Retention',.71,93),('Do Not Track',.97,16)]
paper=[]
for c,f,n in query:paper.append(dict(paper_table='Table 1 query classifier',label=c,reported_f1=f,support=n,pdf_page=7))
segment=[('Data Retention',.71,88),('Data Security',.85,201),('Do Not Track',.95,16),('First Party Collection/Use',.79,1211),('International and Specific Audiences',.95,156),('Introductory/Generic',.70,369),('Policy Change',.88,112),('Non-covered Practice',.70,280),('Privacy Contact Info',.87,137),('Third Party Sharing/Collection',.79,908),('Access, Edit and Deletion',.80,133),('User Choice/Control',.74,433)]
for c,f,n in segment:paper.append(dict(paper_table='Appendix segment classifier',label=c,reported_f1=f,support=n,pdf_page=18))
write('paper_category_results',paper)
pa=[('change-type',.76,18),('notification-type',.71,18),('identifiability',.77,18),('do-not-track-policy',1.,19),('security-measure',.74,19),('personal-information-type',.81,19),('purpose',.83,19),('choice-type',.73,19),('third-party-entity',.73,19),('access-type',.62,19),('audience-type',.97,19),('choice-scope',.59,19),('action-first-party',.65,19),('does-does-not',.86,19),('retention-period',.73,19)]
write('paper_attribute_results',[dict(reported_head=h,reported_average_f1=f,pdf_page=p,note='Published heading retained; third-party-entity table lists action-like values' if h=='third-party-entity' else '') for h,f,p in pa])
aligned=[]
for c,f,n in query:
    row=dict(category=c,paper_table1_f1=f,paper_support=n)
    for m in ['TF-IDF + logistic regression','BERT','RoBERTa','LegalBERT','PrivBERT']:
        r=next(r for r in category if r['condition']=='original' and r['model']==m and r['category']==c);row[m]=r['presence_absence_macro_f1'];row['replication_support']=r['support']
    aligned.append(row)
write('paper_metric_aligned_comparison',aligned)
base={r['condition']:r for r in totals if r['model'].startswith('TF-IDF')};gains=[]
for r in totals:
    if r['model'].startswith('TF-IDF'):continue
    b=base[r['condition']];gains.append(dict(model=r['model'],condition=r['condition'],category_gain=r['category_macro_f1']-b['category_macro_f1'],fine_gain=r['fine_macro_f1']-b['fine_macro_f1']))
write('transformer_gains_over_baseline',gains)
(out/'comparison_summary.json').write_text(json.dumps(dict(totals=totals,gains=gains,all_models_have_identical_eval_rows_and_targets=True),indent=2));print(json.dumps(totals,indent=2))
