import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
# from pcn_model import PCN ## bring in model from museum
## bring in ngc-learn analysis tools
# from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from data.synthetic_data_generator import create_synthetic_data, convert_dataset_to_jax
import torch
from jax import numpy as jnp, random, jit
# from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from pc_time_series import PC_TimeSeries



def train_pc_time_series(dkey, train_dataset, val_dataset, test_dataset, input_dim, output_dim,
              latent_dim=128, batch_size=100, num_epochs=100, lr=1e-3,
              n_inference_steps=20, infer_tau=25
              ):
    ################################################################################
    ## set up JAX seeding
    dkey, *subkeys = random.split(dkey, 10)
    ################################################################################
    _X, _Y = convert_dataset_to_jax(train_dataset)
    Xdev, Ydev = convert_dataset_to_jax(val_dataset)
    Xts, Yts = convert_dataset_to_jax(test_dataset)

    # N = len(_X)
    # n_val = len(_Xdev)
    # m_test = len(_Xts)

    ##################################
    ##################################
    # t = 2

    # _X = _X.reshape(len(_X)//t, t, -1)
    # _Y = _Y.reshape(len(_Y)//t, t, -1)
    # Xdev = _Xdev.reshape(len(_Xdev)//t, t, -1)
    # Ydev = _Ydev.reshape(len(_Ydev)//t, t, -1)
    # Xts = _Xts.reshape(len(_Xts)//t, t, -1)
    # Yts = _Yts.reshape(len(_Yts)//t, t, -1)

    # _X, _Y = _X[:, None, :], _Y[:, None, :]
    # Xdev, Ydev = _Xdev[:, None, :], _Ydev[:, None, :]
    # Xts, Yts = _Xts[:, None, :], _Yts[:, None, :]
    ##################################
    ################################################################################

    ## build model
    model = PC_TimeSeries(subkeys[1], input_dim, output_dim,
                hid1_dim=latent_dim,
                hid2_dim=latent_dim,
                T=n_inference_steps,
                dt=1.,
                batch_size=batch_size,
                tau_m=infer_tau,
                act_fx="sigmoid",
                eta=lr,
                exp_dir="exp",
                model_name="pct"
                )
    sim_start_time = time.time() ## start time profiling
    ################################################################################

    tr_efe, tr_mse = eval_pc_time_series(model, _X, _Y, batch_size=batch_size)
    _, eval_mse = eval_pc_time_series(model, Xdev, Ydev, batch_size=batch_size)
    _, test_mse = eval_pc_time_series(model, Xts, Yts, batch_size=batch_size)
    print("#############################################################")
    print(f"Initial | Train Loss: {tr_mse:.6f} | Val Loss: {eval_mse:.6f} | Test Loss: {test_mse:.6f} | ")
    print("#############################################################")

    n_batches = int(_X.shape[0]/batch_size)
    ################################################################################
    for i in range(num_epochs):
        ## shuffle data (to ensure i.i.d. assumption holds)
        dkey, *subkeys = random.split(dkey, 2)
        ptrs = random.permutation(subkeys[0],_X.shape[0])
        X = _X[ptrs,:]
        Y = _Y[ptrs,:]

        ## begin a single epoch
        train_EFE = 0. ## training free energy (online) estimate
        train_MSE = 0. ## training accuracy score
        n_samp_seen = 0
        train_distance = 0.
        train_Acc = 0.
        #######################################
        for j in range(n_batches):
            dkey, *subkeys = random.split(dkey, 2)
            ## sample mini-batch of patterns
            idx = j * batch_size #j % 2 # 1
            Xb = X[idx: idx + batch_size,:]
            Yb = Y[idx: idx + batch_size,:]

            ## perform a step of inference/learning
            yMu_0, yMu, _EFE, _MSE = model.process(obs=Xb, lab=Yb, adapt_synapses=True)

            train_distance = train_distance + (Yb[:, -1] - yMu_0[:, -1]).sum()
            train_EFE = train_EFE + (_EFE * batch_size)
            train_MSE = train_MSE + _MSE

            n_samp_seen += Yb.shape[0]
            print("\r sample {}/{} ---|\t MSE ={:.4f} |\t EFE ={:.4f}  |\t Y-dist ={:4f}  ".format(
                                        n_samp_seen, len(X),
                                        train_MSE/n_samp_seen,
                                        train_EFE/n_samp_seen,
                                        train_distance/n_samp_seen
            ),
                end="")
        #######################################
        train_mse = (2 * train_MSE)/_X.shape[0]
        eval_distance, eval_mse = eval_pc_time_series(model, Xdev, Ydev, batch_size=batch_size)
        test_distance, test_mse = eval_pc_time_series(model, Xts, Yts, batch_size=batch_size)
        print()
        print(f"Epoch {i} | Train Loss: {train_mse:.4f}  |\t Val Loss: {eval_mse:.4f}  |\t Test Loss: {test_mse:.4f}"
          f" |\t Train dist: {train_distance:.4f} |\t Val dist: {eval_distance:.4f}  |\t Test dist: {test_distance:.4f} ")
    ################################################################################
    print()
    print()
    print("#############################################################")
    print(f"Epoch {i} | Train Loss: {tr_mse:.6f} | Val Loss: {eval_mse:.6f} | Test Loss: {test_mse:.6f} | ")
    print("#############################################################")

    return model, test_mse


def eval_pc_time_series(model, X_eval, Y_eval, batch_size):
    ## evals model's test-time inference performance
    n_batches = int(X_eval.shape[0]/batch_size)

    mse = 0.                    ## mean squared error
    efe = 0.
    acc = 0.
    for j in range(n_batches):
        idx = j * batch_size
        Xb = X_eval[idx: idx + batch_size,:]
        Yb = Y_eval[idx: idx + batch_size,:]
        yMu_0, yMu, _efe, _mse = model.process(obs=Xb, lab=Yb, adapt_synapses=False)

        acc += (Yb[:, -1] - yMu_0[:, -1]).sum()
        # efe = efe + _efe
        mse = mse + _mse

    # efe = efe/(Xdev.shape[0]) ## calc full dev-set EFE
    # mse = mse/(Xdev.shape[0]) ## calc full dev-set MSE

    return acc/X_eval.shape[0], (2 * mse)/X_eval.shape[0]



if __name__ == "__main__":
    # Example usage with synthetic data
    from torch.utils.data import DataLoader, TensorDataset

    from EBM.data.listops_generator import (
        generate_listops_data,
        create_listops_tensors,
        create_vocab
    )

    # Generate ListOps data
    print("\nGenerating ListOps datasets...")
    vocab = create_vocab()
    max_seq_len = 512

    # Generate samples
    train_samples = generate_listops_data(
        num_samples=2000,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=500,
        seed=42
    )

    val_samples = generate_listops_data(
        num_samples=100,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=500,
        seed=43
    )

    test_samples = generate_listops_data(
        num_samples=100,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=500,
        seed=44
    )

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = create_listops_tensors(train_samples, vocab, max_seq_len, device)
    X_val, y_val = create_listops_tensors(val_samples, vocab, max_seq_len, device)
    X_test, y_test = create_listops_tensors(test_samples, vocab, max_seq_len, device)


    y_train = torch.concatenate([X_train[:, 1:], y_train[:, None]], axis=1)
    y_val = torch.concatenate([X_val[:, 1:], y_val[:, None]], axis=1)
    y_test = torch.concatenate([X_test[:, 1:], y_test[:, None]], axis=1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    input_dim = output_dim = X_train.shape[1]

    dkey_ = random.PRNGKey(1234)

    train_pc_time_series(dkey_, train_dataset, val_dataset, test_dataset, input_dim, output_dim)
