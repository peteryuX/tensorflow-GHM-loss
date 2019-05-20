"""TF version for grandient harmonized weights"""

import tensorflow as tf


def GHM_weight(predict, target, valid_mask, bins=10, alpha=0.75,
               dtype=tf.float32, name='GHM_weight'):
    """ Get Grandient Harmonized Weights.
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
    with tf.variable_scope(name):
        _edges = [x / bins for x in range(bins + 1)]
        _edges[-1] += 1e-6
        edges = tf.constant(_edges, dtype=dtype)

        _shape = predict.get_shape().as_list()
        assert len(_shape) == 2
        _init_statistics = (_shape[0] * _shape[1]) / bins
        statistics = tf.get_variable(
            name='statistics', shape=[bins], dtype=dtype, trainable=False,
            initializer=tf.constant_initializer(_init_statistics, dtype=dtype))

        _b_valid = valid_mask > 0
        total = tf.maximum(tf.reduce_sum(tf.cast(_b_valid, dtype=dtype)), 1)

        grandients = tf.abs(predict - target)
        weights = tf.zeros_like(predict)

        # Calculate new statics and new weights
        cond = lambda i, weights_loop, statistics_loop: i < bins
        def body(i, weights_loop, statistics_loop):
            inds = (
                grandients >= edges[i]) & (grandients < edges[i + 1]) & _b_valid
            # number of examples lying in bin, same as R in paper.
            num_in_bin = tf.reduce_sum(tf.cast(inds, dtype=dtype))
            statistics_loop += tf.one_hot(i, bins) * (
                (1 - alpha) * (num_in_bin - statistics_loop))
            grandient_density = statistics_loop[i] * bins
            update_weights = total / grandient_density
            weights_loop = tf.where(inds,
                x=tf.ones_like(weights_loop) * update_weights, y=weights_loop)
            return i + 1, weights_loop, statistics_loop
        _, weights, new_s = tf.while_loop(cond, body, [0, weights, statistics])

        # Avoid the tiny value in statistics
        new_s = tf.maximum(new_s, _init_statistics * 1e-6)
        # Update statistics
        statistics_updated_op = statistics.assign(new_s)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, statistics_updated_op)

    return weights


def main():
    ghm_weights = GHM_weight(predict=tf.constant([[1., 0., 0.5, 0.]]),
                             target=tf.constant([[1., 0., 0., 1.]]),
                             valid_mask=tf.constant([[1., 1., 1., 1.]]))

    # update method same as the batch norm update with optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.constant(0, name='your_optimizer')

    with tf.Session() as sess:
        init_ops = tf.global_variables_initializer()
        sess.run(init_ops)
        
        _, _ghm_weights = sess.run([opt, ghm_weights])
        print('update 1 times: ', _ghm_weights)

        for _ in range(100):
            sess.run([opt])
        print('update 100 times: ', sess.run([ghm_weights]))


if __name__ == '__main__':
    main()
