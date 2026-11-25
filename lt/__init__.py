"""Long-tail toolkit for the TaxoRes-LTIC journal plan (v9.1).

Everything here works on cached frozen-encoder features:
    classes     class order (rarest first, stable), shot groups, R1 asserts
    features    chunked feature store written by tools/extract_features.py
    taxonomy    species -> genus -> family tables
    losses      CE, logit adjustment, Balanced Softmax, LDAM-DRW, LTIC-2/3 branch losses
    heads       linear probe, LTIC / ResLT heads, flat + genus multitask head
    trainer     fast GPU-resident head training with epoch selection on V-tune
    metrics     macro-F1, per-class / per-group accuracy, bootstrap CIs, paired tests
    harness     random search on V-tune, one score on V-select, seeds on test, results CSV
    cf          finding C-F: where logit adjustment's tail false positives come from
    conformal   CP-1 .. CP-4 prediction sets with genus fallback
"""
