# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import numpy as np
import tensorflow as tf


D = 3
ftype = tf.float32
var = tf.zeros(dtype=ftype, shape=[D])
cov = tf.eye(D, dtype=ftype)


def meanf(var):
    return (var + 1.0) ** 2


def lnunnormalp(x, mean, cov):
    x = x - mean
    #     y = tf.tensordot(tf.tensordot(x ,tf.linalg.inv(cov), axes=[1,0]), x, axes=[1,1])
    #     y = tf.diag_part(y)
    d = mean.shape[-1].value
    y = tf.reshape(x, [-1, 1, d]) @ tf.linalg.inv(cov) @ tf.reshape(x, [-1, d, 1])
    # the above one should be the best as broadcasting in multiply, but it is only supported after tf1.13.2
    # for compatibility, use commented tensordot method instead: drawback is hugh memeory consumption
    # see https://github.com/tensorflow/tensorflow/issues/216 and PR therein for broadcasting support on multiply in tf
    return -0.5 * y


def lnnormalp(x, mean, cov):
    k = mean.shape[-1].value
    y = lnunnormalp(x, mean, cov)
    z = 0.5 * tf.constant(k, dtype=ftype) * tf.log(2.0 * np.pi) + 0.5 * tf.log(
        tf.abs(tf.linalg.det(cov))
    )
    return y - z


def fisher1(num_sample=1000, meanf=None, cov=None):
    if meanf is None:
        mean = var
    else:
        mean = meanf(var)
    mgd = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=cov
    )
    r = []
    sample = tf.stop_gradient(mgd.sample(num_sample))
    s = tf.placeholder(dtype=ftype, shape=[None, D])
    lnp = lnnormalp(s, mean, cov)
    dpdv = tf.gradients(lnp, var)
    with tf.Session() as sess:
        sample_ = sess.run(sample)
        sample_ = sample_.reshape(num_sample, 1, 3)
        for i in range(num_sample):
            dpdv_ = sess.run(dpdv, feed_dict={s: sample_[i]})
            r.append(dpdv_[0])
    r = np.array(r)
    fisher = np.zeros([D, D])
    diag = np.mean(r ** 2, axis=0)
    eyemask = np.eye(D, dtype=np.float32)
    fisher += eyemask * diag / 2
    for i in range(D):
        for j in range(i + 1, D):
            fisher[i, j] = np.mean(r[:, i] * r[:, j])
    fisher = fisher + fisher.T

    return fisher


def fisher2(num_sample=1000, meanf=None, cov=None):
    if meanf is None:
        mean = var
    else:
        mean = meanf(var)
    mgd = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=cov
    )
    r = []
    sample = tf.stop_gradient(mgd.sample(num_sample))
    s = tf.placeholder(dtype=ftype, shape=[None, D])
    lnp = lnunnormalp(s, mean, cov)
    dpdv = tf.gradients(lnp, var)
    with tf.Session() as sess:
        sample_ = sess.run(sample)
        sample_ = sample_.reshape(num_sample, 1, 3)
        for i in range(num_sample):
            dpdv_ = sess.run(dpdv, feed_dict={s: sample_[i]})
            r.append(dpdv_[0])
    r = np.array(r)
    fisher = np.zeros([D, D])
    meandpdv = np.mean(r, axis=0)
    diag = np.mean(r ** 2, axis=0) - meandpdv ** 2
    eyemask = np.eye(D, dtype=np.float32)
    fisher += eyemask * diag / 2
    for i in range(D):
        for j in range(i + 1, D):
            fisher[i, j] = np.mean(r[:, i] * r[:, j]) - meandpdv[i] * meandpdv[j]
    fisher = fisher + fisher.T

    return fisher


def fisher3(num_sample=1000, meanf=None, cov=None):
    if meanf is None:
        mean = var
    else:
        mean = meanf(var)
    mgd = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=cov
    )
    sample = tf.stop_gradient(mgd.sample(num_sample))
    lnp = lnnormalp(sample, mean, cov)
    r = tf.reduce_mean(lnp)
    kl = tf.stop_gradient(r) - r
    fisher = tf.hessians(kl, var)
    return fisher


def fisher4(num_sample=1000, meanf=None, cov=None):
    if meanf is None:
        mean = var
    else:
        mean = meanf(var)
    mgd = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=cov
    )
    sample = tf.stop_gradient(mgd.sample(num_sample))
    lnp = lnunnormalp(sample, mean, cov)
    lnpoverp = lnp - tf.stop_gradient(lnp)
    poverp = tf.exp(lnp - tf.stop_gradient(lnp))
    kl = tf.log(tf.reduce_mean(poverp)) - tf.reduce_mean(lnpoverp)
    fisher = tf.hessians(kl, var)
    return fisher


@tf.custom_gradient
def idn(x):
    """
    A wrapper for lnp-tf.stop_gradient(lnp)+1., where lnp is x here
    """

    def grad(dy):
        return dy

    return tf.ones_like(x), grad


def fisher5(num_sample=1000, meanf=None, cov=None):
    if meanf is None:
        mean = var
    else:
        mean = meanf(var)
    mgd = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=cov
    )
    sample = tf.stop_gradient(mgd.sample(num_sample))
    lnp = lnnormalp(sample, mean, cov)
    lnpoverp = idn(lnp)
    kl = -tf.reduce_mean(tf.log(lnpoverp)) / 2
    fisher = tf.hessians(kl, var)
    return fisher


def fisher6(num_sample=1000, meanf=None, cov=None):
    if meanf is None:
        mean = var
    else:
        mean = meanf(var)
    mgd = tf.contrib.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=cov
    )
    sample = tf.stop_gradient(mgd.sample(num_sample))
    lnp = lnunnormalp(sample, mean, cov)
    lnpoverp = idn(lnp)
    kl = tf.log(tf.reduce_mean(lnpoverp)) - tf.reduce_mean(tf.log(lnpoverp))
    fisher = tf.hessians(kl, var)
    return fisher


if __name__ == "__main__":
    f1_ = fisher1(num_sample=1000, meanf=meanf, cov=cov)
    f2_ = fisher2(num_sample=1000, meanf=meanf, cov=cov)
    f3 = fisher3(num_sample=1000, meanf=meanf, cov=cov)
    f4 = fisher4(num_sample=1000, meanf=meanf, cov=cov)
    f5 = fisher5(num_sample=1000, meanf=meanf, cov=cov)
    f6 = fisher6(num_sample=1000, meanf=meanf, cov=cov)

    with tf.Session() as sess:
        [f3_, f4_, f5_, f6_] = sess.run([f3, f4, f5, f6])

    print([f1_, f2_, f3_, f4_, f5_, f6_])  ## 4*eye(D)
