# Leakage audit

Rows: 8,594,200; features: 45

## Duplicates

- n_rows: 8594200
- n_features: 45
- test_rows: 1433369
- test_duplicate_rows: 640806
- test_duplicate_pct: 44.706
- train_rows: 7160831
- train_duplicate_rows: 2663796
- train_duplicate_pct: 37.2
- test_rows_also_in_train: 13551
- test_rows_also_in_train_pct: 0.945
- feature_vectors_with_conflicting_labels: 0
- rows_with_conflicting_labels: 0

## Single-feature shortcuts (top 10, label19)

```
      feature  macro_f1  n_unique_train
          IAT    0.4328           11057
        Srate    0.3657           79209
         Rate    0.3657           79209
      Tot sum    0.3437           28042
Header_Length    0.3373           49034
     Magnitue    0.2867           39525
          AVG    0.2859           39822
          Min    0.2829            7690
     Tot size    0.2806           12636
          Max    0.2606           14165
```

Suggested features to drop (macro-F1 >= 0.95): none

## Row order

Mean lag-1 autocorrelation in file order: 0.282 vs shuffled -0.000. Clearly higher in file order means consecutive flows carry information, which windows use.

## Split inflation (RF)

```
  regime  dedup  n_test  macro_f1_label2  macro_f1_label6  macro_f1_label19
  random  False 1718840           0.9726           0.9297            0.8977
  random   True  980141           0.9720           0.9289            0.8820
official  False 1433369           0.9429           0.8452            0.7441
official   True 1414153           0.9413           0.8441            0.7456
temporal  False 1718835           0.9774           0.9113            0.8504
temporal   True 1558191           0.9762           0.9127            0.8517
```