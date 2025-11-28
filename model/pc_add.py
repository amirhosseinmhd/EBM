"""
Predictive Coding Network for synthetic tasks - CORRECTED VERSION

This implements a hierarchical predictive coding network with:
- Top-down predictions
- Bottom-up error propagation
- Iterative inference through prediction error minimization
- Proper gradient descent for state updates
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_data_generator import create_synthetic_data

from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
# from pcn_model import PCN ## bring in model from museum
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from data.synthetic_data_generator import create_synthetic_data, convert_dataset_to_jax
import torch

from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist

## Main PCN model object
class PCN():
    """
    Structure for constructing the predictive coding network (PCN) in:

    Whittington, James CR, and Rafal Bogacz. "An approximation of the error
    backpropagation algorithm in a predictive coding network with local hebbian
    synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.

    | Node Name Structure:
    | z0 -(W1)-> e1, z1 -(W1)-> e2, z2 -(W3)-> e3;
    | e2 -(E2)-> z1 <- e1, e3 -(E3)-> z2 <- e2
    | Note: W1, W2, W3 -> Hebbian-adapted synapses

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        out_dim: output dimensionality

        hid1_dim: dimensionality of 1st layer of internal neuronal cells

        hid2_dim: dimensionality of 2nd layer of internal neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

        tau_m: membrane time constant of hidden/internal neuronal layers

        act_fx: activation function to use for internal neuronal layers

        exp_dir: experimental directory to save model results

        model_name: unique model name to stamp the output files/dirs with

        save_init: save model at initialization/first configuration time (Default: True)
    """
    def __init__(self, dkey, in_dim=1, out_dim=1, hid1_dim=128, hid2_dim=64, T=10,
                 dt=1., batch_size=1, tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        self.batch_size = batch_size

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        ## hard-coded meta-parameters for this model
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3

        if loadDir is not None:
            ## build from disk
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = RateCell("z0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.z1 = RateCell(
                    "z1", n_units=hid1_dim, tau_m=tau_m, act_fx=act_fx, prior=("gaussian", 0.),
                    integration_type="euler"
                )
                self.e1 = ErrorCell("e1", n_units=hid1_dim)
                self.z2 = RateCell(
                    "z2", n_units=hid2_dim, tau_m=tau_m, act_fx=act_fx, prior=("gaussian", 0.),
                    integration_type="euler"
                )
                self.e2 = ErrorCell("e2", n_units=hid2_dim)
                self.z3 = RateCell("z3", n_units=out_dim, tau_m=0., act_fx="identity")
                self.e3 = ErrorCell("e3", n_units=out_dim)
                ### set up generative/forward synapses
                self.W1 = HebbianSynapse(
                    "W1", shape=(in_dim, hid1_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.W2 = HebbianSynapse(
                    "W2", shape=(hid1_dim, hid2_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[5]
                )
                self.W3 = HebbianSynapse(
                    "W3", shape=(hid2_dim, out_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[6]
                )
                ## set up feedback/error synapses
                self.E2 = StaticSynapse(
                    "E2", shape=(hid2_dim, hid1_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                )
                self.E3 = StaticSynapse(
                    "E3", shape=(out_dim, hid2_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                )

                # ############################################
                # ############################################
                #
                self.z0.batch_size = batch_size
                self.z1.batch_size = batch_size
                self.z2.batch_size = batch_size
                self.z3.batch_size = batch_size
                self.e1.batch_size = batch_size
                self.e2.batch_size = batch_size
                self.e3.batch_size = batch_size
                self.W1.batch_size = batch_size
                self.W2.batch_size = batch_size
                self.W3.batch_size = batch_size
                self.E2.batch_size = batch_size
                self.E3.batch_size = batch_size

                # ############################################
                # ############################################

                ## wire z0 to e1.mu via W1
                self.W1.inputs << self.z0.zF
                self.e1.mu << self.W1.outputs
                self.e1.target << self.z1.z
                ## wire z1 to e2.mu via W2
                self.W2.inputs << self.z1.zF
                self.e2.mu << self.W2.outputs
                self.e2.target << self.z2.z
                ## wire z2 to e3.mu via W3
                self.W3.inputs << self.z2.zF
                self.e3.mu << self.W3.outputs
                self.e3.target << self.z3.z
                ## wire e2 to z1 via W2.T and e1 to z1 via d/dz1
                self.E2.inputs << self.e2.dmu
                self.z1.j << self.E2.outputs
                self.z1.j_td << self.e1.dtarget
                ## wire e3 to z2 via W3.T and e2 to z2 via d/dz2
                self.E3.inputs << self.e3.dmu
                self.z2.j << self.E3.outputs
                self.z2.j_td << self.e2.dtarget
                ## wire e3 to z3 via d/dz3
                #self.z3.j_td << self.e3.dtarget

                ## setup W1 for its 2-factor Hebbian update
                self.W1.pre << self.z0.zF
                self.W1.post << self.e1.dmu
                ## setup W2 for its 2-factor Hebbian update
                self.W2.pre << self.z1.zF
                self.W2.post << self.e2.dmu
                ## setup W3 for its 2-factor Hebbian update
                self.W3.pre << self.z2.zF
                self.W3.post << self.e3.dmu

                ## construct inference / projection model
                self.q0 = RateCell("q0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.q1 = RateCell("q1", n_units=hid1_dim, tau_m=0., act_fx=act_fx)
                self.q2 = RateCell("q2", n_units=hid2_dim, tau_m=0., act_fx=act_fx)
                self.q3 = RateCell("q3", n_units=out_dim, tau_m=0., act_fx="identity")
                self.eq3 = ErrorCell("eq3", n_units=out_dim)
                self.Q1 = StaticSynapse(
                    "Q1", shape=(in_dim, hid1_dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q2 = StaticSynapse(
                    "Q2", shape=(hid1_dim, hid2_dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q3 = StaticSynapse(
                    "Q3", shape=(hid2_dim, out_dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )

                self.q0.batch_size = batch_size
                self.q1.batch_size = batch_size
                self.q2.batch_size = batch_size
                self.q3.batch_size = batch_size
                self.Q1.batch_size = batch_size
                self.Q2.batch_size = batch_size
                self.Q3.batch_size = batch_size
                self.eq3.batch_size = batch_size

                ## wire q0 -(Q1)-> q1, q1 -(Q2)-> q2, q2 -(Q3)-> q3
                self.Q1.inputs << self.q0.zF
                self.q1.j << self.Q1.outputs
                self.Q2.inputs << self.q1.zF
                self.q2.j << self.Q2.outputs
                self.Q3.inputs << self.q2.zF
                self.q3.j << self.Q3.outputs
                self.eq3.mu = self.q3.z
                ## wire q3 to qe3
                # self.eq3.target << self.q3.z

                advance_process = (JaxProcess(name="advance_process")
                                   >> self.E2.advance_state
                                   >> self.E3.advance_state
                                   >> self.z0.advance_state
                                   >> self.z1.advance_state
                                   >> self.z2.advance_state
                                   >> self.z3.advance_state
                                   >> self.W1.advance_state
                                   >> self.W2.advance_state
                                   >> self.W3.advance_state
                                   >> self.e1.advance_state
                                   >> self.e2.advance_state
                                   >> self.e3.advance_state)

                reset_process = (JaxProcess(name="reset_process")
                                 >> self.q0.reset
                                 >> self.q1.reset
                                 >> self.q2.reset
                                 >> self.q3.reset
                                 >> self.eq3.reset
                                 >> self.z0.reset
                                 >> self.z1.reset
                                 >> self.z2.reset
                                 >> self.z3.reset
                                 >> self.e1.reset
                                 >> self.e2.reset
                                 >> self.e3.reset)

                evolve_process = (JaxProcess(name="evolve_process")
                                  >> self.W1.evolve
                                  >> self.W2.evolve
                                  >> self.W3.evolve)

                project_process = (JaxProcess(name="project_process")
                                   >> self.q0.advance_state
                                   >> self.Q1.advance_state
                                   >> self.q1.advance_state
                                   >> self.Q2.advance_state
                                   >> self.q2.advance_state
                                   >> self.Q3.advance_state
                                   >> self.q3.advance_state
                                   >> self.eq3.advance_state)

                processes = (reset_process, advance_process, evolve_process, project_process)

                self._dynamic(processes)

    def _dynamic(self, processes):## create dynamic commands for circuit
        vars = self.circuit.get_components("q0", "q1", "q2", "q3", "eq3",
                                           "Q1", "Q2", "Q3",
                                           "z0", "z1", "z2", "z3",
                                           "e1", "e2", "e3",
                                           "W1", "W2", "W3", "E2", "E3")
        (self.q0, self.q1, self.q2, self.q3, self.eq3, self.Q1, self.Q2, self.Q3,
         self.z0, self.z1, self.z2, self.z3, self.e1, self.e2, self.e3, self.W1,
         self.W2, self.W3, self.E2, self.E3) = vars
        self.nodes = vars

        reset_proc, advance_proc, evolve_proc, project_proc = processes

        self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
        self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
        self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")

        # self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")
        # self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        # self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")
        # self.circuit.add_command(wrap_command(jit(self.circuit.project)), name="project")

        @Context.dynamicCommand
        def clamp_input(x):
            self.z0.j.set(x)
            self.q0.j.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            self.z3.j.set(y)

        @Context.dynamicCommand
        def clamp_infer_target(y):
            self.eq3.target.set(y)
    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.W2.save(model_dir)
            self.W3.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            #self.circuit.save_to_json(self.exp_dir, self.model_name)
    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        print(" > Loading model from ",model_directory)
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)
            processes = (
                self.circuit.reset_process, self.circuit.advance_process,
                self.circuit.evolve_process, self.circuit.project_process
            )
            self._dynamic(processes)
    def process(self, obs, lab, adapt_synapses=True):
        ## can think of the PCN as doing "PEM" -- projection, expectation, then maximization
        eps = 0.001
        lab = jnp.clip(lab, eps, 1. - eps)
        #self.circuit.reset(do_reset=True)
        self.circuit.reset()

        ## pin/tie inference synapses to be exactly equal to the forward ones
        self.Q1.weights.set(self.W1.weights.value)
        self.Q1.biases.set(self.W1.biases.value)
        self.Q2.weights.set(self.W2.weights.value)
        self.Q2.biases.set(self.W2.biases.value)
        self.Q3.weights.set(self.W3.weights.value)
        self.Q3.biases.set(self.W3.biases.value)
        ## pin/tie feedback synapses to transpose of forward ones
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))

        ## Perform P-step (projection step)
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(lab)
        self.circuit.project(t=0., dt=1.) ## do projection/inference

        ## initialize dynamics of generative model latents to projected states
        self.z1.z.set(self.q1.z.value)
        self.z2.z.set(self.q2.z.value)
        ## self.z3.z.set(self.q3.z.value)
        # ### Note: e1 = 0, e2 = 0 at initial conditions
        self.e3.dmu.set(self.eq3.dmu.value)
        self.e3.dtarget.set(self.eq3.dtarget.value)
        ## get projected prediction (from the P-step)
        y_mu_inf = self.q3.z.value
        mse = self.eq3.L.value

        EFE = 0.   ## expected free energy
        # y_mu = 0.
        y_mu = self.e3.mu.value
        if adapt_synapses:
            ## Perform several E-steps
            for ts in range(0, self.T):
                self.circuit.clamp_input(obs) ## clamp data to z0 & q0 input compartments
                self.circuit.clamp_target(lab) ## clamp data to e3.target
                self.circuit.advance(t=ts, dt=1.)

            y_mu = self.e3.mu.value ## get settled prediction
            ## calculate approximate EFE
            L1 = self.e1.L.value
            L2 = self.e2.L.value
            L3 = self.e3.L.value
            EFE = L3 + L2 + L1
            # mse = self.eq3.L.value

            ## Perform (optional) M-step (scheduled synaptic updates)
            if adapt_synapses == True:
                #self.circuit.evolve(t=self.T, dt=self.dt)
                self.circuit.evolve(t=self.T, dt=1.)
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE, mse
    def get_latents(self):
        return self.q2.z.value
    def _get_norm_string(self): ## debugging routine
        _W1 = self.W1.weights.value
        _W2 = self.W2.weights.value
        _W3 = self.W3.weights.value
        _b1 = self.W1.biases.value
        _b2 = self.W2.biases.value
        _b3 = self.W3.biases.value
        _norms = "W1: {} W2: {} W3: {}\n b1: {} b2: {} b3: {}".format(jnp.linalg.norm(_W1),
                                                                      jnp.linalg.norm(_W2),
                                                                      jnp.linalg.norm(_W3),
                                                                      jnp.linalg.norm(_b1),
                                                                      jnp.linalg.norm(_b2),
                                                                      jnp.linalg.norm(_b3))
        return _norms

################################################################################
# print("\nGenerating synthetic datasets...")
# train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
#     task='add',
#     size_train=10000,
#     size_val=256,
#     size_test=256,
#     vec_size=10,
#     device=device
# )




def train_pcn(dkey, train_dataset, val_dataset, test_dataset, input_dim, output_dim,
              latent_dim=128, batch_size=64, num_epochs=100, lr=1e-3,
              n_inference_steps=20, infer_tau=25
              ):
    ################################################################################
    ## set up JAX seeding
    dkey, *subkeys = random.split(dkey, 10)
    ################################################################################
    _X, _Y = convert_dataset_to_jax(train_dataset)
    Xdev, Ydev = convert_dataset_to_jax(val_dataset)
    Xts, Yts = convert_dataset_to_jax(test_dataset)
    ################################################################################

    ## build model
    model = PCN(subkeys[1], input_dim, output_dim,
                hid1_dim=latent_dim,
                hid2_dim=latent_dim,
                T=n_inference_steps,
                dt=1.,
                batch_size=batch_size,
                tau_m=infer_tau,
                act_fx="sigmoid",
                eta=lr,
                exp_dir="exp",
                model_name="pc")
    sim_start_time = time.time() ## start time profiling
    ################################################################################

    tr_efe, tr_mse = eval_pcn(model, _X, _Y, batch_size=batch_size)
    _, eval_mse = eval_pcn(model, Xdev, Ydev, batch_size=batch_size)
    _, test_mse = eval_pcn(model, Xts, Yts, batch_size=batch_size)
    print("#############################################################")
    print(f"Initial | Train Loss: {tr_mse:.6f} | Val Loss: {eval_mse:.6f} | Test Loss: {test_mse:.6f} | ")
    print("#############################################################")

    ################################################################################
    n_batches = int(_X.shape[0]/batch_size)

    for i in range(num_epochs):
        ## shuffle data (to ensure i.i.d. assumption holds)
        dkey, *subkeys = random.split(dkey, 2)
        ptrs = random.permutation(subkeys[0],_X.shape[0])
        X = _X[ptrs,:]
        Y = _Y[ptrs,:]

        ## begin a single epoch
        n_samp_seen = 0
        train_EFE = 0. ## training free energy (online) estimate
        train_MSE = 0. ## training accuracy score
        for j in range(n_batches):
            dkey, *subkeys = random.split(dkey, 2)
            ## sample mini-batch of patterns
            idx = j * batch_size #j % 2 # 1
            Xb = X[idx: idx + batch_size,:]
            Yb = Y[idx: idx + batch_size,:]

            ## perform a step of inference/learning
            yMu_0, yMu, _EFE, _MSE = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
            ## track online training EFE and accuracy
            train_EFE = train_EFE + (_EFE * batch_size)
            train_MSE = train_MSE + _MSE

            # n_samp_seen += Yb.shape[0]
            # if verbosity >= 1:
            #     print("\r  MSE = {} |  EFE = {}  over {} samples ".format(train_MSE/n_samp_seen,
            #                                                          train_EFE/n_samp_seen,
            #                                                          n_samp_seen), end="")

        train_mse = (2 * train_MSE)/_X.shape[0]
        _, eval_mse = eval_pcn(model, Xdev, Ydev, batch_size=batch_size)
        _, test_mse = eval_pcn(model, Xts, Yts, batch_size=batch_size)

        print(f"Epoch {i} | Train Loss: {train_mse:.6f} | Val Loss: {eval_mse:.6f} | Test Loss: {test_mse:.6f} | ")

    ################################################################################

    print()
    print("#############################################################")
    print(f"Epoch {i} | Train Loss: {tr_mse:.6f} | Val Loss: {eval_mse:.6f} | Test Loss: {test_mse:.6f} | ")
    print("#############################################################")

    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time

    return model, test_mse


def eval_pcn(model, X_eval, Y_eval, batch_size):
    ## evals model's test-time inference performance
    n_batches = int(X_eval.shape[0]/batch_size)

    mse = 0.                    ## mean squared error
    efe = 0.
    for j in range(n_batches):
        idx = j * batch_size
        Xb = X_eval[idx: idx + batch_size,:]
        Yb = Y_eval[idx: idx + batch_size,:]
        yMu_0, yMu, _efe, _mse = model.process(obs=Xb, lab=Yb, adapt_synapses=False)

        efe = efe + _efe
        mse = mse + _mse

    # efe = efe/(Xdev.shape[0]) ## calc full dev-set EFE
    # mse = mse/(Xdev.shape[0]) ## calc full dev-set MSE

    return efe/X_eval.shape[0], (2 * mse)/X_eval.shape[0]








class PredictiveLayer(nn.Module):
    """
    Single layer in the predictive coding hierarchy.
    Each layer maintains its own state and computes prediction errors.
    """
    def __init__(self, input_dim, output_dim, activation='silu'):
        super(PredictiveLayer, self).__init__()
        # Top-down prediction weights (from higher layer to this layer)
        self.prediction_weights = nn.Linear(output_dim, input_dim)
        
        # Activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
    def predict_down(self, higher_state):
        """Generate top-down prediction from higher layer."""
        return self.activation(self.prediction_weights(higher_state))

class PredictiveCodingNetwork(nn.Module):
    """
    Hierarchical Predictive Coding Network.
    
    Architecture:
    - Input x -> state1 (input_dim -> latent_dim)
    - state1 -> state2 (latent_dim -> latent_dim)
    - state2 -> state3 (latent_dim -> latent_dim)
    - state3 -> Output (latent_dim -> output_dim)
    
    Energy function:
    E = 0.5 * (||e0||² + ||e1||² + ||e2||² + ||e_output||²)
    
    where:
    - e0 = x - prediction_of_x_from_state1
    - e1 = state1 - prediction_of_state1_from_state2
    - e2 = state2 - prediction_of_state2_from_state3
    - e_output = y_target - output_from_state3
    """
    def __init__(self, input_dim, output_dim, latent_dim=128, activation='silu'):
        super(PredictiveCodingNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Define the hierarchical layers
        # Each layer predicts the layer BELOW it (top-down)
        self.layer1 = PredictiveLayer(input_dim, latent_dim, activation)
        self.layer2 = PredictiveLayer(latent_dim, latent_dim, activation)
        self.layer3 = PredictiveLayer(latent_dim, latent_dim, activation)
        
        # Output layer: state3 -> output
        self.output_layer = nn.Linear(latent_dim, output_dim)
        
        # Store activation for derivative computation
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
    
    def initialize_states(self, batch_size, device):
        """Initialize hidden states for all layers."""
        state1 = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=False)
        state2 = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=False)
        state3 = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=False)
        return state1, state2, state3
    
    def compute_errors(self, x, y_target, state1, state2, state3):
        """
        Compute prediction errors at all layers.
        
        Returns:
            errors: (error0, error1, error2, error_output)
            predictions: (pred0, pred1, pred2, output)
        """
        # Top-down predictions
        pred2 = self.layer3.predict_down(state3)  # state3 predicts state2
        pred1 = self.layer2.predict_down(state2)  # state2 predicts state1
        pred0 = self.layer1.predict_down(state1)  # state1 predicts input x
        
        # Prediction errors
        error2 = state2 - pred2
        error1 = state1 - pred1
        error0 = x - pred0
        
        # Output
        output = self.output_layer(state3)
        if y_target is not None:
            error_output = y_target - output
        else:
            error_output = None
        
        return (error0, error1, error2, error_output), (pred0, pred1, pred2, output)
    
    def activation_derivative(self, x):
        """
        Compute derivative of activation function.
        For common activations, we can compute this efficiently.
        """
        if isinstance(self.activation, nn.Tanh):
            return 1 - torch.tanh(x).pow(2)
        elif isinstance(self.activation, nn.ReLU):
            return (x > 0).float()
        elif isinstance(self.activation, nn.SiLU):
            # SiLU: f(x) = x * sigmoid(x)
            # f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            sigmoid_x = torch.sigmoid(x)
            return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        else:
            return torch.ones_like(x)
    
    def inference_step(self, x, y_target, state1, state2, state3, inference_lr=0.1):
        """
        Single step of predictive coding inference using GRADIENT DESCENT.
        
        Updates states to minimize the total prediction error energy.
        
        Energy: E = 0.5 * (||e0||² + ||e1||² + ||e2||² + ||e_output||²)
        
        Update rule: state_i = state_i - lr * ∂E/∂state_i
        
        Args:
            x: Input data
            y_target: Target output (can be None during unsupervised inference)
            state1, state2, state3: Current states
            inference_lr: Learning rate for inference
        """
        # Compute all errors and predictions
        (error0, error1, error2, error_output), (pred0, pred1, pred2, output) = \
            self.compute_errors(x, y_target, state1, state2, state3)
        
        # Compute pre-activations for derivative
        z1 = self.layer1.prediction_weights(state1)
        z2 = self.layer2.prediction_weights(state2)
        z3 = self.layer3.prediction_weights(state3)
        
        # Activation derivatives
        deriv0 = self.activation_derivative(z1)
        deriv1 = self.activation_derivative(z2)
        deriv2 = self.activation_derivative(z3)
        
        # Gradient of energy with respect to each state
        # ∂E/∂state_i = error_i - (φ'(z_{i-1}) ⊙ error_{i-1}) W_i
        #
        # For batches:
        # - error_i has shape (batch, dim_i)
        # - W_i has shape (dim_{i-1}, dim_i) in the Linear layer's weight matrix
        # - (deriv_{i-1} * error_{i-1}) @ W_i gives shape (batch, dim_i)
        
        # For state1:
        # - Contributes to error1 (being predicted by state2)
        # - Predicts input via error0
        grad_state1 = error1 - (deriv0 * error0) @ self.layer1.prediction_weights.weight
        
        # For state2:
        # - Contributes to error2 (being predicted by state3)
        # - Predicts state1 via error1
        grad_state2 = error2 - (deriv1 * error1) @ self.layer2.prediction_weights.weight
        
        # For state3:
        # - Predicts state2 via error2
        # - Produces output (if y_target provided, also gets supervised signal)
        grad_state3 = -(deriv2 * error2) @ self.layer3.prediction_weights.weight
        
        # Add supervised signal to top layer if target is provided
        if y_target is not None and error_output is not None:
            # Gradient from output layer: W_out^T @ error_output
            # output_layer.weight has shape (output_dim, latent_dim)
            # We need (batch, latent_dim), so: error_output @ output_layer.weight
            grad_state3 = grad_state3 + error_output @ self.output_layer.weight
        
        # GRADIENT DESCENT: minimize energy
        state1_new = state1 - inference_lr * grad_state1
        state2_new = state2 - inference_lr * grad_state2
        state3_new = state3 - inference_lr * grad_state3
        
        # Compute total energy
        total_energy = 0.5 * (
            error0.pow(2).sum(dim=1).mean() + 
            error1.pow(2).sum(dim=1).mean() + 
            error2.pow(2).sum(dim=1).mean()
        )
        if error_output is not None:
            total_energy += 0.5 * error_output.pow(2).sum(dim=1).mean()
        
        return state1_new, state2_new, state3_new, total_energy
    
    def forward(self, x, y_target=None, n_inference_steps=20, inference_lr=0.1):
        """
        Forward pass with iterative inference.
        
        Args:
            x: Input tensor (batch, input_dim)
            y_target: Target output for supervised inference (batch, output_dim)
            n_inference_steps: Number of inference iterations
            inference_lr: Learning rate for inference
        
        Returns:
            output: Predicted output (batch, output_dim)
            final_energy: Final prediction error energy
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        state1, state2, state3 = self.initialize_states(batch_size, device)
        
        # Iterative inference
        for step in range(n_inference_steps):
            state1, state2, state3, total_energy = self.inference_step(
                x, y_target, state1, state2, state3, inference_lr
            )
        
        # Generate output from top layer
        output = self.output_layer(state3)
        
        return output, total_energy
    
    def forward_with_tracking(self, x, y_target=None, n_inference_steps=20, inference_lr=0.1):
        """
        Forward pass with energy tracking (for analysis).
        
        Returns:
            output: Predicted output
            energies: List of total energy at each step
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        state1, state2, state3 = self.initialize_states(batch_size, device)
        
        energies = []
        
        # Iterative inference
        for step in range(n_inference_steps):
            state1, state2, state3, total_energy = self.inference_step(
                x, y_target, state1, state2, state3, inference_lr
            )
            energies.append(total_energy.item())
        
        # Generate output from top layer
        output = self.output_layer(state3)
        
        return output, energies


def train_predictive_coding(train_dataset, val_dataset, test_dataset, input_dim, output_dim,
                            latent_dim=128, batch_size=64, num_epochs=10000, lr=1e-3,
                            n_inference_steps=20, inference_lr=0.1):
    """
    Training loop for predictive coding network.
    
    Two approaches:
    1. Inference with target (supervised): Include y_target during inference
    2. Inference without target, then supervised loss: Run inference freely, then backprop
    
    Here we use approach 1: supervised inference
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Move datasets to device
    x_val, y_val = val_dataset.tensors
    x_test, y_test = test_dataset.tensors
    
    # Model and optimizer
    model = PredictiveCodingNetwork(input_dim, output_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("\nTraining Predictive Coding Network (CORRECTED)")
    print("=" * 60)
    print(f"Architecture: {input_dim} -> {latent_dim} -> {latent_dim} -> {latent_dim} -> {output_dim}")
    print(f"Inference steps per forward pass: {n_inference_steps}")
    print(f"Inference learning rate: {inference_lr}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Sample random batch from training dataset
        indices = torch.randint(0, len(train_dataset), (batch_size,))
        x_batch = train_dataset.tensors[0][indices]
        y_batch = train_dataset.tensors[1][indices]
        
        # Forward pass with inference (INCLUDING target for supervised inference)
        y_pred, energy = model(x_batch, y_target=y_batch, 
                              n_inference_steps=n_inference_steps, 
                              inference_lr=inference_lr)
        
        # Supervised loss on output
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation and test sets
        if epoch % 250 == 0:
            model.eval()
            with torch.no_grad():
                # Validation loss
                y_pred_val, _ = model(x_val, y_target=None, 
                                     n_inference_steps=n_inference_steps,
                                     inference_lr=inference_lr)
                val_loss = criterion(y_pred_val, y_val)
                
                # Test loss
                y_pred_test, _ = model(x_test, y_target=None,
                                      n_inference_steps=n_inference_steps,
                                      inference_lr=inference_lr)
                test_loss = criterion(y_pred_test, y_test)
            
            print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | "
                  f"Test Loss: {test_loss.item():.6f} | Energy: {energy.item():.6f}")
            model.train()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        # Validation set
        y_pred_val, _ = model(x_val, y_target=None,
                             n_inference_steps=n_inference_steps,
                             inference_lr=inference_lr)
        val_loss_final = criterion(y_pred_val, y_val)
        print(f"Final Validation MSE: {val_loss_final.item():.6f}")
        
        # Test set
        y_pred_test, energies = model.forward_with_tracking(
            x_test, y_target=None, n_inference_steps=n_inference_steps, 
            inference_lr=inference_lr
        )
        test_loss_final = criterion(y_pred_test, y_test)
        print(f"Final Test MSE: {test_loss_final.item():.6f}")
        
        # Show inference convergence
        print(f"\n--- Inference Convergence (Test Set) ---")
        print(f"Initial energy: {energies[0]:.6f}")
        print(f"Final energy: {energies[-1]:.6f}")
        print(f"Reduction: {(energies[0] - energies[-1]) / energies[0] * 100:.1f}%")
        
        # Show some examples from test set
        print("\n--- Sample Predictions from Test Set ---")
        for i in range(min(5, len(y_test))):
            pred_str = ', '.join([f"{v:.4f}" for v in y_pred_test[i].cpu().numpy()[:5]])
            true_str = ', '.join([f"{v:.4f}" for v in y_test[i].cpu().numpy()[:5]])
            print(f"Pred: [{pred_str}...], True: [{true_str}...]")
    
    return model


def test_predictive_coding(model, n_inference_steps=50, inference_lr=0.1):
    """Test-time inference with more iterations."""
    print("\nTesting with extended inference...")
    print("=" * 60)
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Generate test data
    vec_size = 10
    n_test = 100
    v1_test = torch.randn(n_test, vec_size, device=device) * 4 - 2
    v2_test = torch.randn(n_test, vec_size, device=device) * 4 - 2
    x_test = torch.cat([v1_test, v2_test], dim=1)
    y_true_test = v1_test + v2_test
    
    model.eval()
    
    with torch.no_grad():
        # Track inference process
        y_pred, energies = model.forward_with_tracking(
            x_test, y_target=None, n_inference_steps=n_inference_steps, 
            inference_lr=inference_lr
        )
        
        print("Inference convergence:")
        for step in [0, 10, 20, 30, 40, n_inference_steps-1]:
            if step < len(energies):
                mse = ((y_pred - y_true_test).pow(2).mean()).item()
                print(f"Step {step:3d} | Energy: {energies[step]:.6f} | MSE: {mse:.6f}")
        
        # Final evaluation
        final_mse = ((y_pred - y_true_test).pow(2).mean()).item()
        print(f"\nFinal Test MSE: {final_mse:.6f}")
        
        # Show example
        print(f"\nExample (first 5 elements):")
        print(f"v1:        {v1_test[0][:5].cpu().numpy()}")
        print(f"v2:        {v2_test[0][:5].cpu().numpy()}")
        print(f"True sum:  {y_true_test[0][:5].cpu().numpy()}")
        print(f"Predicted: {y_pred[0][:5].cpu().numpy()}")
    
    return final_mse


if __name__ == "__main__":
    # Configuration
    task = 'add'
    vec_size = 10
    size_train = 10000
    size_val = 256
    size_test = 256
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate synthetic data
    print("Generating synthetic datasets...")
    train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
        task=task,
        size_train=size_train,
        size_val=size_val,
        size_test=size_test,
        vec_size=vec_size,
        device=device
    )
    print(f"Dataset created: train={size_train}, val={size_val}, test={size_test}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}\n")
    
    # Train predictive coding network
    model = train_predictive_coding(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=128,
        n_inference_steps=20,
        inference_lr=0.05,
        lr=1e-3
    )
    
    # Optional: Test with more inference steps
    test_mse = test_predictive_coding(model, n_inference_steps=50, inference_lr=0.1)
    print("\n" + "=" * 60)
    print(f"Extended Test MSE: {test_mse:.6f}")
    print("=" * 60)