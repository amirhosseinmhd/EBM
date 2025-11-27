from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from pcn_model import PCN ## bring in model from museum
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from synthetic_data_generator import create_synthetic_data, convert_dataset_to_jax
import torch





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


def train_pc(dkey, train_dataset, val_dataset, test_dataset, input_dim, output_dim,
              latent_dim=128, batch_size=64, num_epochs=100, lr=1e-3, n_infer_steps=20, infer_tau_m=25
              ):
    ################################################################################
    ## set up JAX seeding
    dkey, *subkeys = random.split(dkey, 10)

    ################################################################################

    _X, _Y = convert_dataset_to_jax(train_dataset)
    Xdev, Ydev = convert_dataset_to_jax(val_dataset)
    Xts, Yts = convert_dataset_to_jax(test_dataset)

    # input_dim = _X.shape[1]
    # output_dim = _Y.shape[1]


    ################################################################################
    ################################################################################
    ################################################################################

    # # read in general program arguments
    # options, remainder = gopt.getopt(sys.argv[1:], '',
    #                                  ["dataX=", "dataY=", "devX=", "devY=", "verbosity="]
    #                                  )
    # # external dataset arguments
    # dataX = "../../data/mnist/trainX.npy"
    # dataY = "../../data/mnist/trainY.npy"
    # devX = "../../data/mnist/validX.npy"
    # devY = "../../data/mnist/validY.npy"
    # verbosity = 1 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
    # for opt, arg in options:
    #     if opt in ("--dataX"):
    #         dataX = arg.strip()
    #     elif opt in ("--dataY"):
    #         dataY = arg.strip()
    #     elif opt in ("--devX"):
    #         devX = arg.strip()
    #     elif opt in ("--devY"):
    #         devY = arg.strip()
    #     elif opt in ("--verbosity"):
    #         verbosity = int(arg.strip())
    # print("Train-set: X: {} | Y: {}".format(dataX, dataY))
    # print("  Dev-set: X: {} | Y: {}".format(devX, devY))
    #
    # _X = jnp.load(dataX)
    # _Y = jnp.load(dataY)
    # Xdev = jnp.load(devX)
    # Ydev = jnp.load(devY)
    # input_dim = _X.shape[1]
    # patch_shape = (int(jnp.sqrt(input_dim)), int(jnp.sqrt(input_dim)))
    # output_dim = _Y.shape[1]
    #
    #
    # print(input_dim)
    # print(output_dim)
    # input()
    ################################################################################
    ################################################################################
    ################################################################################



    # hid_dim = 128
    # num_epochs = 100
    # batch_size = 64
    n_batches = int(_X.shape[0]/batch_size)
    # verbosity = 1


    ## build model
    print("--- Building Model ---")
    model = PCN(subkeys[1], 
                input_dim, output_dim, hid1_dim=latent_dim, hid2_dim=latent_dim, T=n_infer_steps, dt=1., infer_tau_m=25.,
                act_fx="sigmoid", eta=lr, exp_dir="exp", model_name="pcn"
    )

    def eval_model(model, X, Y, batch_size): ## evals model's test-time inference performance
        n_batches = int(X.shape[0]/batch_size)

        # n_samp_seen = 0
        mse = 0.                                ## mean squared error
        efe = 0.
        for j in range(n_batches):
            idx = j * batch_size
            Xb = X[idx: idx + batch_size,:]
            Yb = Y[idx: idx + batch_size,:]
            yMu_0, yMu, _efe, _mse = model.process(obs=Xb, lab=Yb, adapt_synapses=False)

            efe = efe + _efe
            mse = mse + _mse

            # n_samp_seen += Yb.shape[0]
        # print()
        # input(_efe)

        # efe = efe/(Xdev.shape[0]) ## calc full dev-set EFE
        # mse = mse/(Xdev.shape[0]) ## calc full dev-set MSE

        return efe/X.shape[0], (2 * mse)/X.shape[0]

    # sim_start_time = time.time() ## start time profiling

    tr_efe, tr_mse = eval_model(model, _X, _Y, batch_size=batch_size)
    efe, mse = eval_model(model, Xdev, Ydev, batch_size=batch_size)

    # print("-1: Dev: mse = {}  | Tr: tr_mse = {} ".format(mse/(Xdev.shape[0]),
    #                                                      tr_mse/(_X.shape[0]),
    # ))

    # if verbosity >= 2:
    #     print(model._get_norm_string())
    # trAcc_set.append(tr_acc) ## random guessing is where models typically start
    # acc_set.append(acc)
    # mse_set.append(mse)
    # efe_set.append(-2000.)
    # jnp.save("exp/acc.npy", jnp.asarray(acc_set))
    # jnp.save("exp/efe.npy", jnp.asarray(efe_set))
    # jnp.save("exp/mse.npy", jnp.asarray(mse_set))

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
        _, eval_mse = eval_model(model, Xdev, Ydev, batch_size=batch_size)
        _, test_mse = eval_model(model, Xts, Yts, batch_size=batch_size)

        print(
            f"Epoch {i} | Train Loss: {train_mse:.6f} | Val Loss: {eval_mse:.6f} | Test Loss: {test_mse:.6f} | ")

        # print("{} iters  -- Eval MSE = {}  | Test MSE = {} ".format(i,
        #     (2 * eval_mse) / Xdev.shape[0],
        #     (2 * test_mse) / Xts.shape[0]
        # ))
    # print("------------------------------------")
    # sim_end_time = time.time()
    # sim_time = sim_end_time - sim_start_time
    #
    # print(" Trial.sim_time = {} min == ({} h)  |  ({} sec)".format((sim_time/60.0), (sim_time/3600.0), sim_time))#, vAcc_best))

    return model, eval_mse, test_mse