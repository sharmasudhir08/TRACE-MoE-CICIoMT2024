# Leakage audit

Rows: 8,775,013; features: 45

## Duplicates

- n_rows: 8775013
- n_features: 45
- test_rows: 1614182
- test_duplicate_rows: 724611
- test_duplicate_pct: 44.89
- train_rows: 7160831
- train_duplicate_rows: 2663796
- train_duplicate_pct: 37.2
- test_rows_also_in_train: 13589
- test_rows_also_in_train_pct: 0.842
- feature_vectors_with_conflicting_labels: 0
- rows_with_conflicting_labels: 0

## Single-feature shortcuts (top 10, label19)

```
      feature  macro_f1  n_unique_train
          IAT    0.4926           11057
      Tot sum    0.3574           28042
Header_Length    0.3441           49034
         Rate    0.3288           79209
        Srate    0.3288           79209
     Magnitue    0.3202           39525
          AVG    0.3194           39822
     Tot size    0.3182           12636
          Max    0.2941           14165
          Min    0.2643            7690
```

Suggested features to drop (macro-F1 >= 0.95): none

## Row order

Mean lag-1 autocorrelation in file order: 0.282 vs shuffled -0.000. Clearly higher in file order means consecutive flows carry information, which windows use.

## Split inflation (RF)

```
  regime  dedup  n_test  macro_f1_label2  macro_f1_label6  macro_f1_label19
  random  False 1755003           0.9731           0.9286            0.8958
  random   True  916767           0.9691           0.9200            0.8849
official  False 1614182           0.9428           0.8429            0.8379
official   True 1553969           0.9352           0.8196            0.8277
temporal  False 1754997           0.9775           0.9134            0.8504
temporal   True 1270015           0.9724           0.9009            0.8511
```