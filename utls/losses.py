
import tensorflow as tf


def sigmoid_cross_entropy_balanced(logits, label, name='cross_entrony_loss'):
    """
    Initially proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1.-y)
    count_pos  = tf.reduce_sum(y)
    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


# def class_balanced_cross_entropy_with_logits(logits,label,name='class_ballanced_cross_entropy'):
#
#     # Initialy proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'
#     with tf.name_scope(name) as scope:
#         logits= tf.cast(logits, tf.float32)
#         label = tf.cast(label, tf.float32)
#
#         n_positives = tf.reduce_sum(label)
#         n_negatives = tf.reduce_sum(1.0-label)
#
#         beta = n_negatives/(n_negatives+n_positives)
#         pos_weight = beta / (1-beta)
#         check_weight = tf.identity(beta,name='check')
#
#         cost = tf.nn.weighted_cross_entropy_with_logits(targets=label,logits=logits,pos_weight=pos_weight)
#         loss = tf.reduce_mean((1-beta)*cost)
#
#         return tf.where(tf.equal(beta,1.0),0.0,loss)
