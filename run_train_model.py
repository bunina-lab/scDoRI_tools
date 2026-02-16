def get_training_config(yaml_path=None):
    from scdori import TrainConfig
    return TrainConfig.from_yaml(yaml_path) if yaml_path else TrainConfig()


def main(args):
    import logging
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    from sklearn.preprocessing import OneHotEncoder

    from scdori import (
        load_scdori_inputs,
        save_model_weights,
        set_seed,
        scDoRI,
        train_scdori_phases,
        train_model_grn,
        initialize_scdori_parameters,
        load_best_model,
    )
    import os
    import gc
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    trainConfig = get_training_config(args.config_yaml)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=trainConfig.logging_level)

    if trainConfig.use_wandb:
        try:
            import wandb
            wandb.init(
                project=trainConfig.wandb_project,
                entity=trainConfig.wandb_entity,
                name=trainConfig.wandb_run_name,
                config=trainConfig.__dict__
            )
            logger.info("Weights & Biases initialized.")
        except ImportError:
            logger.warning("wandb not installed, skipping wandb initialization.")

    logger.info("Starting scDoRI pipeline with integrated GRN.")
    set_seed(trainConfig.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    rna_metacell, atac_metacell, gene_peak_dist, insilico_act, insilico_rep = (
        load_scdori_inputs(trainConfig)
    )

    logger.info(f"RNA adata:\n{rna_metacell}")
    logger.info(f"ATAC adata:\n{atac_metacell}")


    gene_peak_fixed = gene_peak_dist.clone()
    gene_peak_fixed[gene_peak_fixed > 0] = 1  # mask for peak-gene links based on distance

    # computing indices of genes which are TFs and setting number of cells per metacell ( set to 1 for single cell data)
    rna_metacell.obs["num_cells"] = 1
    rna_metacell.var["index_int"] = range(rna_metacell.shape[1])
    tf_indices = rna_metacell.var[rna_metacell.var.gene_type == "TF"].index_int.values
    num_cells = rna_metacell.obs.num_cells.values.reshape((-1, 1))

    batch_col = trainConfig.batch_col
    rna_metacell.obs["batch"] = rna_metacell.obs[batch_col].values
    atac_metacell.obs["batch"] = atac_metacell.obs[batch_col].values
    # obtaining onehot encoding for technical batch,

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(rna_metacell.obs["batch"].values.reshape(-1, 1))

    onehot_batch = enc.transform(rna_metacell.obs["batch"].values.reshape(-1, 1)).toarray()

    # 2) Make small train/test sets
    n_cells = rna_metacell.n_obs
    indices = np.arange(n_cells)
    test_size=0.2
    train_idx, eval_idx = train_test_split(indices, test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(train_idx))
    train_loader = DataLoader(
        train_dataset, batch_size=trainConfig.batch_size_cell, shuffle=True, drop_last=rna_metacell.n_obs*(1-test_size) % trainConfig.batch_size_cell < 2

    )

    eval_dataset = TensorDataset(torch.from_numpy(eval_idx))
    eval_loader = DataLoader(
        eval_dataset, batch_size=trainConfig.batch_size_cell, shuffle=False, drop_last=rna_metacell.n_obs*test_size % trainConfig.batch_size_cell < 2
    )

    num_genes = rna_metacell.n_vars
    num_peaks = atac_metacell.n_vars

    num_tfs = insilico_act.shape[1]

    num_batches = onehot_batch.shape[1]

    ## save config file
    trainConfig.save_yaml(Path(trainConfig.data_dir) / "train_config.yaml")

    ## Building the scDoRI model
    model = scDoRI(
        device=device,
        num_genes=num_genes,
        num_peaks=num_peaks,
        num_tfs=num_tfs,
        num_topics=trainConfig.num_topics,
        num_batches=num_batches,
        dim_encoder1=trainConfig.dim_encoder1,
        dim_encoder2=trainConfig.dim_encoder2,
    ).to(device)

    if not os.path.exists(Path(trainConfig.weights_folder_scdori) / "best_scdori_best_eval.pth"):
        # if the model weight file does not exist, then we need to build the model from scratch
    
        ## Phase 1 initialization

        initialize_scdori_parameters(
            model,
            gene_peak_dist.to(device),
            gene_peak_fixed.to(device),
            insilico_act=insilico_act.to(device),
            insilico_rep=insilico_rep.to(device),
            phase="warmup",
        )

        ## Train Phase 1 of scDoRI model
        ## here topics are inferred using reconstruction of ATAC peaks (module 1), reconstruction of RNA from predicted ATAC (module 2) and reconstruction of TF expression (module 3)

        ##Warmup start is used where only module 1 and module 3 are trained for some initial epochs before adding module 2 
        ##
        ## Training Phase 1 of scDoRI model
        logger.info("================================================")
        logger.info("Training Phase 1 of scDoRI model")
        logger.info("================================================")
        model = train_scdori_phases(
            model,
            device,
            train_loader,
            eval_loader,
            rna_metacell,
            atac_metacell,
            num_cells,
            tf_indices,
            onehot_batch,
            trainConfig,
        )

        # saving the model weight correspoinding to final epoch where model stopped training
        save_model_weights(model, Path(trainConfig.weights_folder_scdori), "scdori_final")
    else:
        # loading the best checkpoint from Phase 1
        logger.info("Loading best checkpoint from Phase 1")
        logger.info(f"Loading from {Path(trainConfig.weights_folder_scdori) / 'best_scdori_best_eval.pth'}")

        model = load_best_model(
            model, Path(trainConfig.weights_folder_scdori) / "best_scdori_best_eval.pth", device
    )

    ## Phase 2 training and saving model weights

    # train Phase 2 of scDoRI model, TF-gene links are learnt in this phase and used to reconstruct gene-expression profiles
    logger.info("================================================")
    logger.info("Training Phase 2 of scDoRI model")
    logger.info("================================================")
    model = train_model_grn(
        model,
        device,
        train_loader,
        eval_loader,
        rna_metacell,
        atac_metacell,
        num_cells,
        tf_indices,
        onehot_batch,
        trainConfig,
    )

    # saving the model weight correspoinding to final epoch where model stopped training
    save_model_weights(model, Path(trainConfig.weights_folder_grn), "scdori_final")

    trainConfig.save_yaml(Path(trainConfig.data_dir) / "train_config.yaml")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-ome scDoRI training pipeline.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to YAML configuration file. If not provided, default config is used.",
    )
    args = parser.parse_args()
    main(args)