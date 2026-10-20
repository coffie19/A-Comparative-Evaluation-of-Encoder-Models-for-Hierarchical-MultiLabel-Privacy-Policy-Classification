# A simple baseline changes the transformer comparison

## Technical summary

The trained TF-IDF + logistic regression baseline obtains **0.7282 category Macro-F1 and 0.4499 fine-grained Macro-F1** on the same 1,440 evaluation segments as the four transformer replications. PrivBERT leads at category classification (0.7947), but the simple baseline exceeds PrivBERT's fine-grained score (0.2268) by 0.2231 and leads on 17 of twenty attribute heads.

After the previously defined 22-segment duplicate/near-match exclusion, the baseline retains 0.4474 fine-grained Macro-F1 versus PrivBERT's 0.2259. The simple baseline therefore changes the substantive conclusion: transformers help category recognition in these implementations, but the current transformer fine-grained results do not establish a benefit over sparse linear modelling.

The original Polisis paper reports 0.84 for its query-category classifier and 0.81 for its segment-category classifier under different label and scoring conventions. Its results are included below as historical reference, with a metric-aligned diagnostic rather than an asserted like-for-like reproduction.

## Matched evaluation contract

All five locally evaluated models use the original 65 training / 50 evaluation policy split, 2,352 training segments, 1,440 evaluation segments, ten parent categories and 138 attribute values across twenty heads. Annotation rows are unioned into the original segment targets. Category Macro-F1 here is the mean positive-label F1 over ten categories. Fine-grained Macro-F1 is the mean over all 138 values, scoring each head only on reference-applicable segments and setting zero-denominator F1 to zero.

Predicted categories gate attribute heads using the same shared-head mapping as rematch. Attribute classifiers are trained only on applicable training segments. Label vocabulary is frozen from the outer training policies. The original and filtered evaluation rows and reference targets are identical across all five local models; no reported paper score enters this matched comparison.

The baseline uses full segment text. Transformers use their original 384-token limits; one evaluation segment per model is truncated. This small input-policy difference is disclosed rather than silently changing checkpoint inference.

## A trained baseline with selection confined to training policies

Features combine word unigrams/bigrams and character-within-word 3–5 grams, with sublinear term frequency, minimum document frequency two, and at most 60,000 features per branch. Each branch uses its own L2 normalization. The final feature matrix has 48,269 columns.

Each label has an independent L2-regularized logistic regression using liblinear and balanced binary class weights. Constant predictors handle any training subset with only one class. Three GroupKFold splits partition the 65 training policies; validation documents never contribute to the corresponding feature vocabulary or IDF. The outer training-derived output vocabulary is held fixed across these folds.

Regularization C was selected from 1, 4 and 16 by the mean of category and fine-grained Macro-F1, averaged across the three policy folds. **C=1** was selected, then fitted on all 65 training policies. The evaluation set was not used for selection. The probability threshold stays at 0.5; no test-set threshold tuning was performed. Logistic solver seed is 1729; this is one final baseline fit, not a multi-seed experiment.

The baseline and transformers have different training procedures: balanced logistic weights and independent fits versus the saved joint transformer training with smoothed positive weights. Therefore this experiment compares these complete implementations; it does not isolate the encoder architecture as the cause of the difference.

## Transformers improve categories, but lose on attributes

Relative to the baseline, category Macro-F1 gains are +0.0153 for BERT, +0.0368 for RoBERTa, +0.0332 for LegalBERT and +0.0665 for PrivBERT. Fine-grained Macro-F1 changes are negative for all four: the strongest transformer checkpoint, PrivBERT, is 0.2231 below the baseline.

Read the following plots separately: category and attribute Macro-F1 average different output vocabularies. The evidence supports a category-level advantage for these transformer checkpoints and a strong sparse-linear result for attribute prediction. It does not support claiming that this benchmark requires a transformer.

PrivBERT has the highest category score; the baseline supplies a meaningful lower-complexity reference.

The baseline has the highest fine-grained score by a large margin; evaluating only transformer-to-transformer differences would miss this result.

## The fine-grained advantage spans most attribute heads

The baseline exceeds PrivBERT on seventeen of twenty heads. Examples include purpose (0.6671 versus 0.1286), personal-information type (0.5671 versus 0.1016), security measure (0.3696 versus 0.1028) and choice type (0.5448 versus 0.2751). PrivBERT retains advantages on does/does not, collection mode and do-not-track policy.

This strengthens the need to diagnose transformer attribute training, class weighting and threshold behaviour rather than treating the low fine-grained score as evidence of unavoidable task difficulty. The baseline still performs poorly on some detailed heads, so the result does not imply that the hierarchy is solved.

## The result survives duplicate filtering

The shared sensitivity cohort excludes exact and normalized cross-split matches, collapses the remaining normalized duplicate within evaluation, and excludes near candidates at the audit's 0.90 cosine cutoff. It contains 1,418 segments. The baseline's fine-grained score is 0.4474, still 0.2215 above PrivBERT. The overall local rankings remain unchanged.

These are post-hoc exclusion results using frozen predictions, not retraining on deduplicated data. Near candidates include shared templates whose meanings can differ, so this cohort is a robustness check rather than an adjudicated clean benchmark.

## The original paper uses different metrics and label sets

Harkous et al., *Polisis: Automated Analysis and Presentation of Privacy Policies Using Deep Learning*, USENIX Security 2018, reports CNN classifiers with privacy-specific word embeddings. Table 1 (PDF page 7; proceedings page 536) reports average F1 **0.84** for the nine-label query-category classifier. The appendix (PDF page 18; proceedings page 547) reports **0.81** for a twelve-label segment classifier, including three distinct Other-related labels.

Footnote 3 on PDF page 6 defines F1 by averaging performance on both the presence and absence of each label. Our previous rematch headline metric averages positive-label F1. The paper's 0.84 and 0.81 must therefore not be directly subtracted from rematch's positive-label Macro-F1.

Both workflows name a 65/50 policy split, but matching counts do not establish matching policy identities or aggregation. Published positive supports also differ substantially: for example, First Party Collection has support 1,267 in Table 1 and 1,211 in the appendix, versus 594 in our held-out targets. The supplied paper does not establish an identical split/target contract for these runs. The local transformer checkpoint evaluations are rematch replications; they are **not an exact reimplementation of Polisis's original CNN**.

## A metric-aligned diagnostic is useful, but not an exact paper replication

For each local model, I recomputed the mean of positive-class and negative-class F1 for each of the nine non-Other categories, then averaged across those categories. This follows the presence/absence averaging described in the paper's footnote. The resulting means are **0.8416 baseline, 0.8526 BERT, 0.8638 RoBERTa, 0.8625 LegalBERT and 0.8815 PrivBERT**, alongside the paper's reported Table 1 average of 0.84.

The table aligns category names and the averaging operation, while displaying different positive supports. It does not retrain our ten-label segment classifiers as the paper's nine-label query classifier, recover its original policy identities, or resolve aggregation differences. Similar scores are therefore not evidence of successful exact replication or superiority over the original paper.

## Published attribute results are retained without inventing a global score

The appendix gives averages for fifteen selected attribute classifiers, not a directly comparable global Macro-F1 over our 138 values. Their reported values are preserved below. The vocabularies differ: for example, the paper shows two change-type values while rematch has five. The paper's table headed third-party-entity lists action-like values; that heading is retained with an explicit note rather than silently remapped.

Consequently, these historical averages document the original findings but cannot serve as a matched measure of transformer gain or loss. The local head-by-head table above is the valid matched comparison.

## Validation and limits

Reloading the saved logistic model reproduces all 1,440 exported probability vectors within 1e-12. Independent scikit-learn F1 calculations reproduce the baseline's category and fine-grained results. All five local exports have identical evaluation keys and reference labels. CV fold policies are disjoint, and the original transformer metrics reproduce their saved summaries.

The experiment does not estimate multi-seed variability or causal architectural effects. Baseline hyperparameter selection and class weighting differ from the original transformer runs. External base-model pretraining contamination, template dependence across policies and historical manual model selection remain unresolved. Training-policy CV prevents using the fifty held-out policies for this baseline's parameter selection, but does not make the pre-existing test set newly independent of the research process.

## Revised conclusion and next experiment

The simple baseline materially changes the conclusion: PrivBERT is strongest among the tested systems for coarse categories, while TF-IDF + logistic regression is strongest for the implemented fine-grained task. A blanket claim that transformers are needed, or that their overall capability follows from their category ranking, is unsupported.

Next, compare class weights, per-head fitting and validation-selected thresholds under matched policy folds; then repeat all methods on duplicate-grouped splits and matched seeds. An exact Polisis replication would additionally require its original split identities, annotation aggregation, twelve-category targets, attribute vocabularies and CNN training procedure.

## Further questions

Does the sparse model's attribute advantage persist when transformer loss weighting and validation tuning are matched? Which baseline gains come from explicit lexical cues versus repeated templates? Can the original Polisis evaluation contract be recovered well enough for an exact replication?