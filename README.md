ADMC
==========

*Infinite order automatic differentiation for Monte Carlo expectations from unnormalized probability distributions.*

### Introduction

Due to the nature of Metropolis-Hasting algorithm, we can simulate the distribution by Monte Carlo as long as we have the knowledge of the ratio between probabilities (densities) for two different configurations. Namely, we can sample data from unnormalized probability distributions with unknown normalized factors of the distribution (which usually denoted as partition function in statistics physics). There are various scenarios for such MC with unormalized probability, including MCMC to estimate posteriors in Bayesian inference context and classical Monte Carlo as well as Quantum Monte Carlo methods to evaluate observable quantities from Hamiltonian models in statistical physics context.

The method to compute the derivatives of such MC expectation from unnormalized probability is lack in the literature. To utilize the power of existing ML frameworks, the only thing to hack is the object function. According to our papers, infinitely AD-aware MC expectation objective is:

<p align="center">
  <img width="130" src="./static/admcobj.svg"/>
</p>

where p is the unnormalized probability (density).

And we have the following examples to show the power of this new ADMC technique.

### Examples

Note the code experiments are all implemented with TensorFlow (tested on tf1.14, both cpu and gpu version) in static computational graph mode.

* Fast locate the critical temperature for 2D Ising model

  In this example, we utilize various features that ML frameworks enable us. We implement Wolff update scheme for 2D Ising model with vectorize consideration so that tens of thounds of Markov Chains can be simulated at the same time. Together with GPU acceleration, automatic differentiation infrastructure and carefully designed optimizers, ML frameworks can make our life easy even beyond ML tasks.

<p align="center">
	<img width="550" src="./static/tctraining.svg"/>
</p>

* Calculate Fisher information matrix for unnormalized distribution with AD approach on KL divergence

  In this example, we show several approaches to calculate Fisher matrix for normalized and unnormalized distributions. The fancy AD approach to evaluate Fisher matrix is to compute KL divergence defined below in the forward pass. And the Hessian of such KL object is just Fisher information matrix.

    <p align="center">
       <img width="450" src="./static/klobj.svg"/>
    </p>

* End-to-end, easy-to-implement VMC with neural network wavefunctions

  Forget about all techniques and derivations for VMC in terms of its gradients or SR methods. We only need to minimize the energy, and we have no interest to derive the gradients for energy expectation by hand. Why bother when you are equipped with AD infrastructure. With moderate effort and buitin optimizer Adam, you can achieve state-of-the-art result on 2D Heisenberg model. Implementation of VMC has never been such easy as this.
  
<p align="center">
	<img width="550" src="./static/vmcworkflow.svg"/>
</p>

<p align="center">
	<img width="550" src="./static/walltime.svg"/>
</p>

## Reference

For details on ADMC theory and application, see our work: 