import json,csv,datetime,sqlite3,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,required=True);args=parser.parse_args();OUT=args.results_dir;COMP=OUT/'comparison'
r=json.loads((OUT/'results.json').read_text());s=json.loads((COMP/'comparison_summary.json').read_text())
def read(name):
    rows=list(csv.DictReader((COMP/(name+'.csv')).open(encoding='utf-8-sig')))
    for row in rows:
        for k,v in row.items():
            try:row[k]=float(v)
            except ValueError:pass
    return rows
ds={'overall':[x for x in s['totals'] if x['condition']=='original'],'filtered':[x for x in s['totals'] if x['condition']=='duplicate_filtered'],'cv':r['cv'],'paper_categories':read('paper_category_results'),'paper_attributes':read('paper_attribute_results'),'aligned':read('paper_metric_aligned_comparison')}
attributes=read('attribute_comparison');categories=read('category_comparison');models=[x['model'] for x in ds['overall']]
ds['heads']=[];ds['categories']=[]
for field,source,dest,metric in [('head',attributes,'heads','macro_f1'),('category',categories,'categories','positive_f1')]:
    for label in dict.fromkeys(x[field] for x in source):
        row={field:label}
        for m in models:row[m]=next(x[metric] for x in source if x[field]==label and x['model']==m and x['condition']=='original')
        ds[dest].append(row)
sources=[];blocks=[];tables=[];charts=[]
def md(id,body):blocks.append(dict(id=id,type='markdown',body=body))
def table(id,title,fields):
    rows=ds[id];cols=list(rows[0]);filename=id+'_report.csv'
    with (COMP/filename).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE results ('+','.join('"'+k+'"' for k in cols)+')');db.executemany('INSERT INTO results VALUES ('+','.join('?' for k in cols)+')',[[x[k] for k in cols] for x in rows]);assert len(db.execute('SELECT * FROM results').fetchall())==len(rows)
    sources.append(dict(id=id,label=title,path='comparison/'+filename,query=dict(sql='SELECT * FROM results',language='sql',engine='SQLite',tables_used=['results'],description='Exact retrieval of comparison rows. Local metrics are recomputed from actual predictions; paper rows are transcribed from the supplied Harkous et al. PDF, pages 7, 18 and 19, visually checked against the tables.')))
    tables.append(dict(id=id,title=title,dataset=id,sourceId=id,columns=[dict(field=k,label=k.replace('_',' '),type='text' if k in ['model','condition','category','head','paper_table','label','reported_head','note'] else 'number') for k in fields]));blocks.append(dict(id=id+'_table',type='table',tableId=id))
title='A simple baseline changes the transformer comparison'
md('title','# '+title)
md('summary','''## Technical summary

The trained TF-IDF + logistic regression baseline obtains **0.7282 category Macro-F1 and 0.4499 fine-grained Macro-F1** on the same 1,440 evaluation segments as the four transformer replications. PrivBERT leads at category classification (0.7947), but the simple baseline exceeds PrivBERT's fine-grained score (0.2268) by 0.2231 and leads on 17 of twenty attribute heads.

After the previously defined 22-segment duplicate/near-match exclusion, the baseline retains 0.4474 fine-grained Macro-F1 versus PrivBERT's 0.2259. The simple baseline therefore changes the substantive conclusion: transformers help category recognition in these implementations, but the current transformer fine-grained results do not establish a benefit over sparse linear modelling.

The original Polisis paper reports 0.84 for its query-category classifier and 0.81 for its segment-category classifier under different label and scoring conventions. Its results are included below as historical reference, with a metric-aligned diagnostic rather than an asserted like-for-like reproduction.''')
md('scope','''## Matched evaluation contract

All five locally evaluated models use the original 65 training / 50 evaluation policy split, 2,352 training segments, 1,440 evaluation segments, ten parent categories and 138 attribute values across twenty heads. Annotation rows are unioned into the original segment targets. Category Macro-F1 here is the mean positive-label F1 over ten categories. Fine-grained Macro-F1 is the mean over all 138 values, scoring each head only on reference-applicable segments and setting zero-denominator F1 to zero.

Predicted categories gate attribute heads using the same shared-head mapping as rematch. Attribute classifiers are trained only on applicable training segments. Label vocabulary is frozen from the outer training policies. The original and filtered evaluation rows and reference targets are identical across all five local models; no reported paper score enters this matched comparison.

The baseline uses full segment text. Transformers use their original 384-token limits; one evaluation segment per model is truncated. This small input-policy difference is disclosed rather than silently changing checkpoint inference.''')
md('method',f'''## A trained baseline with selection confined to training policies

Features combine word unigrams/bigrams and character-within-word 3–5 grams, with sublinear term frequency, minimum document frequency two, and at most 60,000 features per branch. Each branch uses its own L2 normalization. The final feature matrix has {r['feature_count']:,} columns.

Each label has an independent L2-regularized logistic regression using liblinear and balanced binary class weights. Constant predictors handle any training subset with only one class. Three GroupKFold splits partition the 65 training policies; validation documents never contribute to the corresponding feature vocabulary or IDF. The outer training-derived output vocabulary is held fixed across these folds.

Regularization C was selected from 1, 4 and 16 by the mean of category and fine-grained Macro-F1, averaged across the three policy folds. **C={r['selected_C']:g}** was selected, then fitted on all 65 training policies. The evaluation set was not used for selection. The probability threshold stays at 0.5; no test-set threshold tuning was performed. Logistic solver seed is 1729; this is one final baseline fit, not a multi-seed experiment.

The baseline and transformers have different training procedures: balanced logistic weights and independent fits versus the saved joint transformer training with smoothed positive weights. Therefore this experiment compares these complete implementations; it does not isolate the encoder architecture as the cause of the difference.''')
table('cv','Training-policy cross-validation',list(ds['cv'][0]))
md('overall','''## Transformers improve categories, but lose on attributes

Relative to the baseline, category Macro-F1 gains are +0.0153 for BERT, +0.0368 for RoBERTa, +0.0332 for LegalBERT and +0.0665 for PrivBERT. Fine-grained Macro-F1 changes are negative for all four: the strongest transformer checkpoint, PrivBERT, is 0.2231 below the baseline.

Read the following plots separately: category and attribute Macro-F1 average different output vocabularies. The evidence supports a category-level advantage for these transformer checkpoints and a strong sparse-linear result for attribute prediction. It does not support claiming that this benchmark requires a transformer.''')
table('overall','Matched original evaluation scores',['model','n','category_macro_f1','fine_macro_f1','oracle_fine_macro_f1'])
for field,titlepart in [('category_macro_f1','Category Macro-F1'),('fine_macro_f1','Fine-grained Macro-F1')]:
    cid='chart_'+field;charts.append(dict(id=cid,title=titlepart,type='bar',intent='comparison',question='How much do transformer checkpoints gain over a simple fitted baseline?',rationale='Five models share identical evaluation examples and scoring rules; separate bars avoid mixing output vocabularies.',dataset='overall',sourceId='overall',encodings={'x':{'field':'model','type':'nominal'},'y':{'field':field,'type':'quantitative'}},valueFormat='number',layout='full'));blocks.append(dict(id=cid+'_block',type='chart',chartId=cid))
    md(cid+'_reading','PrivBERT has the highest category score; the baseline supplies a meaningful lower-complexity reference.' if field.startswith('category') else 'The baseline has the highest fine-grained score by a large margin; evaluating only transformer-to-transformer differences would miss this result.')
md('heads','''## The fine-grained advantage spans most attribute heads

The baseline exceeds PrivBERT on seventeen of twenty heads. Examples include purpose (0.6671 versus 0.1286), personal-information type (0.5671 versus 0.1016), security measure (0.3696 versus 0.1028) and choice type (0.5448 versus 0.2751). PrivBERT retains advantages on does/does not, collection mode and do-not-track policy.

This strengthens the need to diagnose transformer attribute training, class weighting and threshold behaviour rather than treating the low fine-grained score as evidence of unavoidable task difficulty. The baseline still performs poorly on some detailed heads, so the result does not imply that the hierarchy is solved.''')
table('heads','Attribute-head comparison',['head']+models)
table('categories','Positive-label F1 by category',['category']+models)
md('duplicates','''## The result survives duplicate filtering

The shared sensitivity cohort excludes exact and normalized cross-split matches, collapses the remaining normalized duplicate within evaluation, and excludes near candidates at the audit's 0.90 cosine cutoff. It contains 1,418 segments. The baseline's fine-grained score is 0.4474, still 0.2215 above PrivBERT. The overall local rankings remain unchanged.

These are post-hoc exclusion results using frozen predictions, not retraining on deduplicated data. Near candidates include shared templates whose meanings can differ, so this cohort is a robustness check rather than an adjudicated clean benchmark.''')
table('filtered','Matched duplicate-filtered evaluation',['model','n','category_macro_f1','fine_macro_f1'])
md('paper','''## The original paper uses different metrics and label sets

Harkous et al., *Polisis: Automated Analysis and Presentation of Privacy Policies Using Deep Learning*, USENIX Security 2018, reports CNN classifiers with privacy-specific word embeddings. Table 1 (PDF page 7; proceedings page 536) reports average F1 **0.84** for the nine-label query-category classifier. The appendix (PDF page 18; proceedings page 547) reports **0.81** for a twelve-label segment classifier, including three distinct Other-related labels.

Footnote 3 on PDF page 6 defines F1 by averaging performance on both the presence and absence of each label. Our previous rematch headline metric averages positive-label F1. The paper's 0.84 and 0.81 must therefore not be directly subtracted from rematch's positive-label Macro-F1.

Both workflows name a 65/50 policy split, but matching counts do not establish matching policy identities or aggregation. Published positive supports also differ substantially: for example, First Party Collection has support 1,267 in Table 1 and 1,211 in the appendix, versus 594 in our held-out targets. The supplied paper does not establish an identical split/target contract for these runs. The local transformer checkpoint evaluations are rematch replications; they are **not an exact reimplementation of Polisis's original CNN**.''')
table('paper_categories','Original paper category results',list(ds['paper_categories'][0]))
md('aligned','''## A metric-aligned diagnostic is useful, but not an exact paper replication

For each local model, I recomputed the mean of positive-class and negative-class F1 for each of the nine non-Other categories, then averaged across those categories. This follows the presence/absence averaging described in the paper's footnote. The resulting means are **0.8416 baseline, 0.8526 BERT, 0.8638 RoBERTa, 0.8625 LegalBERT and 0.8815 PrivBERT**, alongside the paper's reported Table 1 average of 0.84.

The table aligns category names and the averaging operation, while displaying different positive supports. It does not retrain our ten-label segment classifiers as the paper's nine-label query classifier, recover its original policy identities, or resolve aggregation differences. Similar scores are therefore not evidence of successful exact replication or superiority over the original paper.''')
table('aligned','Paper Table 1 and metric-aligned local diagnostic',list(ds['aligned'][0]))
md('paper_attributes','''## Published attribute results are retained without inventing a global score

The appendix gives averages for fifteen selected attribute classifiers, not a directly comparable global Macro-F1 over our 138 values. Their reported values are preserved below. The vocabularies differ: for example, the paper shows two change-type values while rematch has five. The paper's table headed third-party-entity lists action-like values; that heading is retained with an explicit note rather than silently remapped.

Consequently, these historical averages document the original findings but cannot serve as a matched measure of transformer gain or loss. The local head-by-head table above is the valid matched comparison.''')
table('paper_attributes','Original paper selected attribute averages',list(ds['paper_attributes'][0]))
md('validation','''## Validation and limits

Reloading the saved logistic model reproduces all 1,440 exported probability vectors within 1e-12. Independent scikit-learn F1 calculations reproduce the baseline's category and fine-grained results. All five local exports have identical evaluation keys and reference labels. CV fold policies are disjoint, and the original transformer metrics reproduce their saved summaries.

The experiment does not estimate multi-seed variability or causal architectural effects. Baseline hyperparameter selection and class weighting differ from the original transformer runs. External base-model pretraining contamination, template dependence across policies and historical manual model selection remain unresolved. Training-policy CV prevents using the fifty held-out policies for this baseline's parameter selection, but does not make the pre-existing test set newly independent of the research process.''')
md('next','''## Revised conclusion and next experiment

The simple baseline materially changes the conclusion: PrivBERT is strongest among the tested systems for coarse categories, while TF-IDF + logistic regression is strongest for the implemented fine-grained task. A blanket claim that transformers are needed, or that their overall capability follows from their category ranking, is unsupported.

Next, compare class weights, per-head fitting and validation-selected thresholds under matched policy folds; then repeat all methods on duplicate-grouped splits and matched seeds. An exact Polisis replication would additionally require its original split identities, annotation aggregation, twelve-category targets, attribute vocabularies and CNN training procedure.''')
md('questions','''## Further questions

Does the sparse model's attribute advantage persist when transformer loss weighting and validation tuning are matched? Which baseline gains come from explicit lexical cues versus repeated templates? Can the original Polisis evaluation contract be recovered well enough for an exact replication?''')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=now,blocks=blocks,charts=charts,tables=tables,sources=sources),snapshot=dict(version=1,generatedAt=now,status='ready',datasets=ds),sources=sources)
(OUT/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8');(OUT/'report.md').write_text('\n\n'.join(b['body'] for b in blocks if b['type']=='markdown'),encoding='utf-8')
