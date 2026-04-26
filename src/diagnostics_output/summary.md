# HERMES — Error Diagnostics (Full Pipeline (4 SECs))

**File:** `results_current.json`  
**Format:** Full Pipeline (4 SECs)  
**Classes:** positive, negative, neutral  
**Total evaluated:** 8922 (skipped: 47)  
**Correct:** 5151  
**Errors:** 3771  

## Metrics

| Metric | Value |
|--------|-------|
| AvgRec | 0.6633 |
| Macro-F1 | 0.5688 |
| F1^PN | 0.6554 |
| Accuracy | 0.5773 |
| Micro-F1 | 0.5773 |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Gold | Pred |
|-------|-----------|--------|----|------|------|
| positive | 0.4926 | 0.8301 | 0.6183 | 1842 | 3104 |
| negative | 0.5634 | 0.8986 | 0.6925 | 2780 | 4434 |
| neutral | 0.8121 | 0.2614 | 0.3955 | 4300 | 1384 |

## Confusion Matrix

| Gold \ Pred | Positive | Negative | Neutral |
|------------|-------:|-------:|-------:|
| positive | 1529 | 159 | 154 |
| negative | 176 | 2498 | 106 |
| neutral | 1399 | 1777 | 1124 |

## Error Distribution

| Pattern | Count | % of Total Errors | File |
|---------|-------|------------------:|------|
| neutral>negative | 1777 | 47.1% | `errors_neutral_to_negative.md` |
| neutral>positive | 1399 | 37.1% | `errors_neutral_to_positive.md` |
| negative>positive | 176 | 4.7% | `errors_negative_to_positive.md` |
| positive>negative | 159 | 4.2% | `errors_positive_to_negative.md` |
| positive>neutral | 154 | 4.1% | `errors_positive_to_neutral.md` |
| negative>neutral | 106 | 2.8% | `errors_negative_to_neutral.md` |

## Resource Usage

| Metric | Total | Avg per Tweet |
|--------|------:|--------------:|
| Input tokens | 80,339,416 | 9,005 |
| Output tokens | 18,785,928 | 2,106 |
| Total tokens | 99,125,344 | 11,110 |
| Time (seconds) | 127,475.9 | 14.3 |
