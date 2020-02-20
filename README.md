# [TensorFlow GHM loss weights](https://github.com/peteryuX/tensorflow-GHM-loss)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peteryuX/tensorflow-GHM-loss/blob/master/notebooks/colab-github-demo.ipynb)
![Star](https://img.shields.io/github/stars/peteryuX/tensorflow-GHM-loss)
![Fork](https://img.shields.io/github/forks/peteryuX/tensorflow-GHM-loss)
![License](https://img.shields.io/github/license/peteryuX/tensorflow-GHM-loss)

This is a simple tensorflow implementation of the loss weights in **Gradient Harmonized Single-stage Detector** published on **AAAI 2019 Oral**.

Original Paper (Arxiv): [Link](https://arxiv.org/abs/1811.05181)

****

## Proposed GHM Function

You can get the GHM weights by the ***get_ghm_weight()*** function in ***tf_ghm_loss.py***.
And use these weights to modify your loss term like theory in paper.
The brief information of this function as below:

```python
def get_ghm_weight(predict, target, valid_mask, bins=10, alpha=0.75,
               dtype=tf.float32, name='GHM_weight'):
    """ Get gradient Harmonized Weights.
    This is an implementation of the GHM ghm_weights described
    in https://arxiv.org/abs/1811.05181.
    Args:
        predict:
            The prediction of categories branch, [0, 1].
            -shape [batch_num, category_num].
        target:
            The target of categories branch, {0, 1}.
            -shape [batch_num, category_num].
        valid_mask:
            The valid mask, is 0 when the sample is ignored, {0, 1}.
            -shape [batch_num, category_num].
        bins:
            The number of bins for region approximation.
        alpha:
            The moving average parameter.
        dtype:
            The dtype for all operations.
    
    Returns:
        weights:
            The beta value of each sample described in paper.
    """
```

****

## Toy Demo :coffee:

The demo state like bellow:
- prediction: [1., 0., 0.5, 0.]
- target:     [1., 0., 0., 1.]

You can find more details in [tf_ghm_loss.py](https://github.com/peteryuX/tensorflow-GHM-loss/blob/master/tf_ghm_loss.py).

### Run
```
python tf_ghm_loss.py
```

### Output
```
update 1 times:  [[0.5        0.5        0.72727275 0.72727275]]
update 100 times:  [array([[0.20000002, 0.20000002, 0.40000004, 0.40000004]], dtype=float32)]
```

****

## Relevant materials :beer:

- https://arxiv.org/abs/1811.05181
    - Gradient Harmonized Single-stage Detector

- https://github.com/libuyu/GHM_Detection
    - Torch implementation of “Gradient Harmonized Single-stage Detector”.
