"""
Python3 example on how to implement end-to-end ADVMC in TensorFlow.
The specific example here is 2D Heisenberg model.
"""

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tensorflow as tf
import numpy as np


class ADVMC:
    def __init__(self, size, log_phi, n_mc, session=None, ftype=tf.float32):
        self.SR_flag = 0
        self.history = []
        self.n_mc = n_mc
        self.ftype = ftype
        self.learning_rate = 0.001
        self.size = size
        self.log_phi = log_phi
        self.learning_rate_placeholder = tf.placeholder(ftype, [], name="learning_rate")
        self.s0 = tf.Variable(tf.zeros([n_mc, size]), ftype)
        self.log_phi0 = log_phi(self.s0)
        self.s1, self.log_phi1, self.accept_ratio = self.update()
        self.h = self.h_loc()
        self.g = self.idn(2 * self.log_phi0)
        self.energy = tf.reduce_mean(self.g * self.h) / tf.reduce_mean(self.g)
        self.KL_divergence = tf.math.log(tf.reduce_mean(self.g)) - tf.reduce_mean(
            tf.math.log(self.g)
        )  ## only valid in SR mode
        self.s_update = tf.assign(self.s0, self.s1)
        self.grad = tf.gradients(self.energy, self.log_phi.trainable_variables)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_placeholder)
        self.train = self.optimizer.apply_gradients(
            zip(self.grad, self.log_phi.trainable_variables)
        )
        if session is None:
            self.sess = tf.Session()
        else:
            self.sess = session
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    @tf.custom_gradient
    def idn(x):
        def grad(*dy):
            return dy

        return tf.ones_like(x), grad

    def h_loc(self):
        raise Exception("no implementation in base class ADVMC")

    def update(self):
        # only shows spin flip update preserving total spin sz,
        # the most general case is not implemented here
        s0 = self.s0
        mask = s0
        o = tf.stop_gradient(tf.random.categorical(tf.math.log(mask), 1)[:, 0])
        eo = tf.one_hot([o], self.size)[0]
        mask = 1 - s0
        o = tf.stop_gradient(tf.random.categorical(tf.math.log(mask), 1)[:, 0])
        eo = eo + tf.one_hot([o], self.size)[0]
        ds = eo - 2 * eo * s0
        s1 = self.s0 + ds
        log_phi1 = self.log_phi(s1)
        flag = tf.cast(
            tf.less(
                (tf.random.uniform([self.n_mc])),
                tf.math.exp((log_phi1 - self.log_phi0) * 2),
            ),
            self.ftype,
        )
        log_phi1 = log_phi1 * flag + self.log_phi0 * (1.0 - flag)
        accept = tf.reduce_mean(flag)
        flag = tf.reshape(flag, [1, self.n_mc])
        flag = tf.transpose(tf.tile(flag, [self.size, 1]))
        s1 = s0 + flag * ds
        return s1, log_phi1, accept

    def set_state(self, state):
        self.sess.run(tf.assign(self.s0, state))

    def set_learning_rate(self, r):
        self.learning_rate = r

    def set_optimizer(self, opt, **kw):
        if opt == "SR":
            if self.SR_flag == 0:
                self.SR_flag = 1  ## SR related graph added
                epsilon = kw["epsilon"]

                def FIM_F(g, X):
                    hessian = tf.hessians(self.KL_divergence, X)
                    X_size = tf.size(X)
                    hessian = tf.reshape(hessian, [X_size, X_size]) + epsilon * tf.eye(
                        X_size
                    )
                    FIM = tf.linalg.inv(hessian)
                    return tf.reshape(
                        tf.linalg.matvec(FIM, tf.reshape(g, [-1])), tf.shape(X)
                    )

                ## only consider block diagonal approximation on FIM, no coupling terms between different layer
                F = [
                    FIM_F(g, x)
                    for g, x in zip(self.grad, self.log_phi.trainable_variables)
                ]
                self.optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate_placeholder
                )
                self.train = self.optimizer.apply_gradients(
                    zip(F, self.log_phi.trainable_variables)
                )
            else:
                print("Using existing SR optimizer node")
        else:
            temp = set(tf.all_variables())
            self.optimizer = opt(self.learning_rate_placeholder, **kw)
            self.train = self.optimizer.apply_gradients(
                zip(self.grad, self.log_phi.trainable_variables)
            )
            self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    def update_configuration(self):
        acc, _ = self.sess.run(
            [self.accept_ratio, self.s_update],
            feed_dict={self.learning_rate_placeholder: self.learning_rate},
        )
        print("Acc Ratio: {:<10.2%}".format(acc))

    def update_param(self, n_t):
        for j in range(n_t):
            self.sess.run(
                self.s_update,
                feed_dict={self.learning_rate_placeholder: self.learning_rate},
            )
        energy, accept, _ = self.sess.run(
            [self.energy, self.accept_ratio, self.train],
            feed_dict={self.learning_rate_placeholder: self.learning_rate},
        )
        energy = energy / self.size

        return energy, accept

    def optimize(self, N, n_t, debug=True):
        for i in range(N):
            energy, accept = self.update_param(n_t)

            if debug:
                print(
                    "Epoch:{:^4}".format(i),
                    "Energy: {:<10.5f}".format(energy),
                    "Accept Ratio: {:<10.2%}".format(accept),
                    "Learning Rate: {:<10.5f}".format(self.learning_rate),
                )
            self.history.append(energy)


class Heisenberg_2d(ADVMC):
    def __init__(self, lx, ly, log_phi, n_mc, session=None, ftype=tf.float32):
        self.oh = []
        onehot = np.zeros([1, lx * ly])
        for i in range(0, lx * ly):
            onehot[0, i] = 1.0
            self.oh.append(tf.constant(onehot, ftype))
            onehot[0, i] = 0.0
        self.lx = lx
        self.ly = ly
        super().__init__(lx * ly, log_phi, n_mc, session=session, ftype=ftype)
        self.set_state(
            tf.tile(
                tf.reshape(
                    tf.transpose(
                        tf.stack(
                            [
                                tf.ones([int(self.size / 2)]),
                                tf.zeros([int((self.size + 1) / 2)]),
                            ]
                        )
                    ),
                    [1, self.size],
                ),
                [n_mc, 1],
            )
        )  ## Sz=0 state as the initial condition

    def h_loc(self):
        J = 1.0
        oh = self.oh
        lx = self.lx
        ly = self.ly
        n_mc = self.n_mc
        ftype = self.ftype

        def code(i, j):
            return (i % lx) * ly + j % ly

        phi0 = self.log_phi(self.s0)
        Hj = tf.zeros([n_mc], ftype)
        Ht = tf.zeros([n_mc], ftype)
        for i in range(0, lx):
            for j in range(0, ly):
                Hj = (
                    Hj
                    + J
                    * (2 * self.s0[:, code(i, j)] - 1)
                    * (2 * self.s0[:, code(i + 1, j)] - 1)
                    + J
                    * (2 * self.s0[:, code(i, j)] - 1)
                    * (2 * self.s0[:, code(i, (j + 1))] - 1)
                )

        for i in range(0, lx):
            for j in range(0, ly):

                ss = self.s0
                eo = oh[code(i, j)]
                ss = eo - 2 * eo * ss + ss

                eo = oh[code(i + 1, j)]
                sx = eo - 2 * eo * ss + ss
                Ht = Ht - tf.math.abs(
                    sx[:, code(i, j)] - sx[:, code(i + 1, j)]
                ) * 2.0 * J * tf.math.exp(self.log_phi(sx) - phi0)

                eo = oh[code(i, j + 1)]
                sy = eo - 2 * eo * ss + ss
                Ht = Ht - tf.math.abs(
                    sy[:, code(i, j)] - sy[:, code(i, j + 1)]
                ) * 2.0 * J * tf.math.exp(self.log_phi(sy) - phi0)

        return tf.stop_gradient(Hj + Ht) / 4.0


def test():  ## user case on how to utilize the general ADVMC class above
    class Log_phi(tf.keras.Model):
        def __init__(self, l, name=None):
            super().__init__(name=name)
            initializer = tf.keras.initializers.RandomUniform(
                minval=-0.2, maxval=0.2, seed=None
            )

            self.dense0 = tf.keras.layers.Dense(
                4 * l,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer=initializer,
            )
            self.dense1 = tf.keras.layers.Dense(
                4 * l,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer=initializer,
            )
            self.dense2 = tf.keras.layers.Dense(
                4 * l,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer=initializer,
            )
            self.dense3 = tf.keras.layers.Dense(
                1, kernel_initializer=initializer, bias_initializer=initializer
            )

        def call(self, x):
            nn = self.dense0(x)
            nn = self.dense1(nn)
            nn = self.dense2(nn)
            nn = self.dense3(nn)
            return tf.reduce_sum(nn, axis=-1)

    lx = 4
    ly = 4
    log_phi = Log_phi(lx * ly)
    admc = Heisenberg_2d(lx, ly, log_phi, 50)
    admc.set_learning_rate(0.002)
    # admc.set_optimizer(tf.train.AdamOptimizer)
    admc.optimize(60, lx * ly)

    # simple visualization
    import matplotlib.pyplot as plt

    data = admc.history
    plt.plot(data, label="VMC")
    plt.plot(-0.7017802 * np.ones(np.stack(data).shape), "r--", label="exact")
    plt.ylabel("E")
    plt.xlabel("Number of iterations")
    plt.legend(loc="upper right", shadow=True, fontsize="small")
    plt.savefig("result.pdf", format="pdf")


if __name__ == "__main__":
    test()
