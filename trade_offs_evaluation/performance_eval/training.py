"""
Util file to evaluate performance of fl clients and ol servers
"""

try:
    from mpc_dpsgd_trainer import DP_Trainer, DP_TrainerEncrypted
    from models import CryptenThreeLayerNN, ThreeLayerNN
except ImportError:
    from utils.mpc_dpsgd_trainer import DP_Trainer, DP_TrainerEncrypted
    from utils.models import CryptenThreeLayerNN, ThreeLayerNN


import torch
import crypten.communicator as comm
import logging
import crypten

def dpsgd_training(
    x_train,
    y_train,
    x_val = None,
    y_val = None,
    y_val_onehot = None,
    val_metric = "accuracy",
    noise_type = "local",
    experiment_name = "dpsgd",
    collusion_multiplier = 1,
    batch_size = 128,
    epochs = 2,
    lr = 0.01,
    eps = 10,
    clipping_threshold = 3.0,
    delta = 1e-6,
    batched = False,
    verbose = False,
    device = "cpu",
    validate = False,
    sampling_rate = 1.0,
    subsampling_type = "poisson",
    non_dp = False,
    optimizer_type = "dpsgd",
    num_labels = 62,
    model = CryptenThreeLayerNN(),
    n_parties = 2,
):
    
    model.train()
    model.encrypt()

    trainer = DP_TrainerEncrypted(
        model = model,
        num_labels=num_labels,
        batch_size=batch_size,
        num_epochs=epochs,
        loss_fn=crypten.nn.CrossEntropyLoss(),
        lr=lr,
        epsilon=eps,    
        clipping_threshold=clipping_threshold,
        optimizer_type=optimizer_type,
        collusion_multiplier=collusion_multiplier,
        noise_type=noise_type,
        delta=delta,
        experiment_name=experiment_name,
        verbose=verbose,
        device=device,
        sampling_rate=sampling_rate,
        subsampling_type=subsampling_type,
    )

    logging.info("Encrypting data party 0")
    x_train_0 = crypten.cryptensor(x_train[:x_train.size(0)//2], src=0)
    logging.info("Encrypting data party 1")
    x_train_1 = crypten.cryptensor(x_train[x_train.size(0)//2:], src=1 if n_parties > 1 else 0)

    logging.info("Concatenating encrypted data")
    x_train_enc = crypten.cat([x_train_0, x_train_1], dim=0)
    #x_train_enc = crypten.cryptensor(torch.tensor([]))
    #step_size = 50000
    #n_steps = math.ceil(x_train.size(0) / step_size)
    #n_steps = 4
    #step_size = x_train.size(0) // n_steps
    #for i in range(n_steps, step_size):
    #    x_train_enc = crypten.cat([x_train_enc, crypten.cryptensor(x_train[i:i+step_size])], dim=0)

    #x_train_enc = crypten.cryptensor(x_train)
    logging.info(f"x_train shape = {x_train_enc.shape}")
    logging.info("Encrypting labels")
    y_train_enc = crypten.cryptensor(y_train)
    comm.get().reset_communication_stats()
    logging.info("Starting training")

    if non_dp:
        losses = trainer.train_non_dp(
            x=x_train_enc,
            y=y_train_enc,
            x_val = None,
            y_val = None,
            validate=validate,
            eval_one_batch=True,
            test = False,
            batched=batched,
        )
        return {
            "losses": [(l.get_plain_text() if isinstance(l, crypten.CrypTensor) else l).tolist() for l in losses],
            "timing": trainer.timing.to_dict(),
        }

    if validate:
        x_val_enc = crypten.cryptensor(x_val)
        losses, val_scores, val_losses = trainer.train_and_validate(
            x=x_train_enc,
            y=y_train_enc,
            x_val=x_val_enc,
            y_val=y_val,
            y_val_onehot=y_val_onehot,
            validation_metric=val_metric,
            model_name=experiment_name,
            batched=batched,
        )

        comm_stats = comm.get().get_communication_stats()
        logging.info(f"Communication stats: {comm_stats}")

        return {
            "losses": [l.tolist() for l in losses],
            "val_scores": val_scores,
            "val_losses" : [l.tolist() for l in val_losses],
            "timing": trainer.timing.to_dict(),
            "comm_stats": comm_stats,
        }
    
    if batched:
        losses = trainer.train_batched(
            x=x_train_enc,
            y=y_train_enc,
        )
    else:
        losses = trainer.train(
            x=x_train_enc,
            y=y_train_enc,
        )

    comm_stats = comm.get().get_communication_stats()
    logging.info(f"Communication stats: {comm_stats}")

    return {
        "losses": [l.tolist() for l in losses],
        "timing": trainer.timing.to_dict(),
        "comm_stats": comm_stats,
    }


def clear_training(
    x_train,
    y_train,
    x_val = None,
    y_val = None,
    x_test = None,
    y_test = None,
    val_metric = "accuracy",
    noise_type = "local",
    experiment_name = "dpsgd",
    collusion_multiplier = 1,
    batch_size = 128,
    epochs = 2,
    lr = 0.01,
    eps = 10,
    clipping_threshold = 3.0,
    delta = 1e-6,
    batched = False,
    verbose = False,
    device = "cpu",
    validate = True,
    sampling_rate = 1.0,
    subsampling_type = "approx",
    optimizer_type = "dpsgd",
    non_dp = False,
    test = False,
    model_path = None,
    eval_one_batch = False,
    test_device = "mps",
    num_labels = 10,
    model = ThreeLayerNN(),
):

    model.train()
    loss = torch.nn.CrossEntropyLoss()

    trainer = DP_Trainer(
        model = model,
        num_labels=num_labels,
        batch_size=batch_size,
        num_epochs=epochs,
        loss_fn=loss,
        lr=lr,
        epsilon=eps,
        clipping_threshold=clipping_threshold,
        optimizer_type=optimizer_type,
        collusion_multiplier=collusion_multiplier,
        noise_type=noise_type,
        delta=delta,
        experiment_name=experiment_name,
        verbose=verbose,
        device=device,
        sampling_rate=sampling_rate,
        subsampling_type=subsampling_type,
        #model_path=model_path,
        #test_device=test_device,
    )

    if non_dp:
        losses = trainer.train_non_dp(
            x=x_train,
            y=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            validate=validate,
            test=test,
            eval_one_batch=eval_one_batch,
            batched=batched,
        )
        return {
            "losses": [l.tolist() for l in losses],
            "timing": trainer.timing.to_dict(),
        }

    if validate:
        losses, val_scores, val_losses = trainer.train_and_validate(
            x=x_train,
            y=y_train,
            x_val=x_val,
            y_val=y_val,
            validation_metric=val_metric,
            model_name=experiment_name,
            batched=batched,
        )
        if test:
            test_scores = trainer.test(
                x=x_test,
                y=y_test,
            )
            return {
                "losses": [l.tolist() for l in losses],
                "val_scores": val_scores,
                "val_losses" : [l.tolist() for l in val_losses],
                "test_scores": test_scores,
                "timing": trainer.timing.to_dict(),
            }

        return {
            "losses": [l.tolist() for l in losses],
            "val_scores": val_scores,
            "val_losses" : [l.tolist() for l in val_losses],
            "timing": trainer.timing.to_dict(),
        }
    
    losses = trainer.train(
        x=x_train,
        y=y_train,
        batched=batched,
        #test=test,
    )
    if test:
        losses, test_scores = losses

    return {
        "noise_stddev": trainer.noise_stddev,
        "losses": [l.tolist() for l in losses],
        "test_scores": test_scores if test else None,
        "timing": trainer.timing.to_dict(),
    }

    