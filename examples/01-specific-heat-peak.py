"""
Python3 script for fast searching on phase boundary.
Showcase example is transition in 2D Ising model.
"""

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import numpy as np
import tensorflow as tf
from functools import partial

itype = tf.int32
ftype = tf.float32

# spin shape (batch, x, y) where batch is for different Markov chains
# find boundary say for each batch there is some 1
# mask is new comer, as Fold, one only track neighbor of Fold on each round
# cluster is a tracker of old stuff to be flipped
# note: spin is consisting of -1 and 1 while mask/cluster is consiting of 0 and 1
def boundary(cluster, mask):  # has n in sourronding with n neighbors but itself is 0
    with tf.name_scope("boundary") as scope:
        sl, sr, su, sd = (
            tf.roll(mask, -1, -1),
            tf.roll(mask, 1, -1),
            tf.roll(mask, -1, -2),
            tf.roll(mask, 1, -2),
        )
        maskt = mask + sl + sr + su + sd
        maskt = maskt * (1 - cluster)
        return maskt


# expand the cluster in each step
def expand(p, state, spin, cluster, mask):
    with tf.name_scope("expand") as scope:
        boundary_mask = tf.cast(
            boundary(cluster, mask), dtype=ftype
        )  # counts the neighbors
        spin1 = tf.cast(tf.abs((spin + state) / 2), dtype=ftype)
        spin1 = spin1 * boundary_mask
        spin1 = tf.cast(
            tf.less(
                tf.random.uniform(spin1.shape, maxval=1.0, dtype=ftype),
                1 - (1 - p) ** spin1,
            ),
            dtype=itype,
        )
        return cluster + spin1, spin1  # new cluster and new boundary


def expand_cond(cluster0, mask0):
    return tf.greater(tf.reduce_sum(mask0), tf.constant(0))


def wolff(p, spin):
    with tf.name_scope("wolff") as scope:
        pos = tf.random.uniform(
            dtype=itype,
            shape=spin.shape[:-2],
            maxval=spin.shape[-2] * spin.shape[-1],
            minval=0,
        )
        mask0 = tf.reshape(
            tf.one_hot(pos, depth=spin.shape[-2] * spin.shape[-1], dtype=itype),
            shape=spin.shape,
        )
        cluster0 = mask0
        state = tf.reduce_sum(spin * mask0, axis=[-1, -2], keepdims=True)
        expand_partial = partial(expand, p, state, spin)
        cluster0, mask0 = tf.while_loop(expand_cond, expand_partial, [cluster0, mask0])
        spin = spin * (1 - cluster0) - spin * cluster0
        return spin


def mc(temp=2.3, J=-1.0, times=100, spin=None, measure=None):
    p = 1 - tf.exp(2.0 / temp * J)
    if spin is None:
        spin = [10, 10, 10]
    if isinstance(spin, list):
        #         spin = tf.random.uniform(shape=spin, dtype=tf.int32, maxval=2, minval=0) * 2 - 1
        spin = tf.ones(dtype=itype, shape=spin)
    obj = []
    param = {"temp": temp, "J": J}
    for i in range(times):
        spin = wolff(p, spin)
        obj.append(measure(spin, param))
    return tf.stack(obj), spin


def mc_derivative(temp=2.3, J=-1.0, times=100, burnin=10, spin=None, return_conf=False):
    if isinstance(spin, list):
        size = spin[-1] * spin[-2]
    else:
        size = spin.shape[-1].value * spin.shape[-2].value
    en, conf = mc(temp=temp, J=J, times=times, spin=spin, measure=energy_d)
    l = tf.reduce_mean(en[burnin:], axis=[0, -1])
    obj = l[0] / l[1] / tf.constant(size, dtype=ftype)
    if return_conf:
        return obj, conf
    return obj


def measure(spin, param):
    J = kws["J"]
    e = energy(spin, J)
    return e


def energy(spin, param):  # PBC
    J = param["J"]
    sl, su = tf.roll(spin, -1, -1), tf.roll(spin, -1, -2)
    e = tf.cast(spin * (sl + su), dtype=ftype) * J
    e = tf.reduce_sum(e, axis=[-1, -2])
    return e


def absmag(spin, **kws):
    return tf.abs(tf.reduce_sum(spin, axis=[-1, -2]))


# note measure functions cannot be dressed with defun somehow in eager mode
# since high order derivative will have problems in practice
def energy_d(spin, param):
    J, temp = param["J"], param["temp"]
    e = energy(spin, param)
    beta = 1.0 / temp
    poverp = tf.exp(-beta * e + tf.stop_gradient(beta * e))
    return tf.stack([e * poverp, poverp])


def ddwrapper(temp=2.3, J=-1.0, times=100, burnin=10, spin=None):
    if (not isinstance(temp, tf.Tensor)) and (not isinstance(temp, tf.Variable)):
        temp = tf.constant(temp)

    ene, spin = mc_derivative(
        temp=temp, J=J, times=times, burnin=burnin, spin=spin, return_conf=True
    )
    cv = tf.gradients(ene, temp)
    dcvdt = tf.gradients(cv, temp)
    return ene, cv, dcvdt, spin


if __name__ == "__main__":

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    temp = tf.Variable(initial_value=2.3)
    sshape = [
        50,
        10,
        10,
    ]  # 2D Ising model with lattice size 10*10 using 50 Markov chains
    spin_ = np.ones(dtype=np.int32, shape=sshape)
    spin = tf.placeholder(dtype=itype, shape=sshape)
    epochs = 50
    times = 10
    burnin = 0
    templ = []
    #     p = tf.print(spin)
    #     with tf.control_dependencies([p]):
    e, cv, dcvdt, spinc = ddwrapper(temp, -1.0, times, burnin, spin)
    opt = optimizer.apply_gradients([[-dcvdt[0], temp]])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            print("epoch:", i)
            t = sess.run(temp)
            print(t)
            templ.append(t)
            [e_, cv_, dcvdt_, spin_, _] = sess.run(
                [e, cv, dcvdt, spinc, opt], feed_dict={spin: spin_}
            )
            print(cv_, dcvdt_)

    stable_steps = int(epochs / 3)
    print(np.mean(templ[stable_steps:]))  # estimated transition temperature
