# TensorFlow GHM loss weights

:fire: This is a simple tensorflow implementation of the loss weights in [Gradient Harmonized Single-stage Detector](https://arxiv.org/abs/1811.05181) published on AAAI 2019 (**Oral**). :fire:

### Toy Demo :coffee:

Run
```
python tf_ghm_loss.py
```

Output
```
update 1 times:  [[0.5        0.5        0.72727275 0.72727275]]
update 100 times:  [array([[0.20000002, 0.20000002, 0.40000004, 0.40000004]], dtype=float32)]
```

### Relevant materials :beer:

- https://arxiv.org/abs/1811.05181
    - Gradient Harmonized Single-stage Detector

- https://github.com/libuyu/GHM_Detection
    - Torch implementation of “Gradient Harmonized Single-stage Detector”.
