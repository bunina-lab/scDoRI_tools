from typing import Literal
import yaml
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Convert matrix to long format (source-target-weight)
def matrix_to_edge_list(matrix, weight_threshold=0):
    """
    Convert TF-gene matrix to edge list format.
    
    Parameters:
    -----------
    matrix : pd.DataFrame
        Matrix with genes as rows and TFs as columns
    weight_threshold : float
        Minimum weight value to include (default: 0, includes all non-zero)
    
    Returns:
    --------
    pd.DataFrame with columns: source, target, weight
    """
    # Stack the matrix to get all combinations
    stacked = matrix.stack()
    
    # Filter out zero or below-threshold weights
    stacked = stacked[abs(stacked) > weight_threshold]
    
    # Reset index to get source and target columns
    edge_list = stacked.reset_index()
    edge_list.columns = ['target', 'source', 'weight']
    
    # Reorder columns to match your desired format
    edge_list = edge_list[['source', 'target', 'weight']]
    
    # Reset index
    edge_list = edge_list.reset_index(drop=True)
    
    return edge_list

def get_celltype_GRN(grn_topic, df_topic_celltype, tf_names, gene_names, outdir):

    GRN = np.tensordot(df_topic_celltype.values.T, grn_topic, axes=([1], [0]))

    for idx, cell_ in enumerate(df_topic_celltype.columns.to_list()):

        cell_GRN = GRN[idx, : , :]
        cell_GRN_df = pd.DataFrame(cell_GRN.T, index=gene_names, columns=tf_names, dtype=np.float32)
        
        outpath = os.path.join(outdir, f"{cell_}_GRN.tsv".replace(" ", "_").replace("/", "_"))
        cell_GRN_df.to_csv(outpath, sep="\t", index=True)
        matrix_to_edge_list(cell_GRN_df, weight_threshold=0).to_csv(outpath.replace(".tsv", "_edge_list.tsv"), sep="\t", index=False)
    
    return GRN

def get_celltype_eGRN(GRN_celltype, gene_peak_tensor, df_topic_celltype, peak_names, tf_names, outdir):

    eGRN = np.tensordot(GRN_celltype, gene_peak_tensor, axes=([2], [0]))

    for idx, cell_ in enumerate(df_topic_celltype.columns.to_list()):

        cell_GRN = eGRN[idx, : , :]
        cell_GRN_df = pd.DataFrame(cell_GRN.T, index=peak_names, columns=tf_names, dtype=np.float32)
        
        outpath = os.path.join(outdir, f"{cell_}_eGRN.tsv".replace(" ", "_").replace("/", "_"))
        cell_GRN_df.to_csv(outpath, sep="\t", index=True)
        matrix_to_edge_list(cell_GRN_df, weight_threshold=0).to_csv(outpath.replace(".tsv", "_edge_list.tsv"), sep="\t", index=False)

    return eGRN

def save_json(dict_obj, save_path):
    import json
    
    with open(save_path, "w") as fh:
        json.dump(dict_obj, fh)

def get_training_config(yaml_path=None):
    from scdori import TrainConfig
    return TrainConfig.from_yaml(yaml_path) if yaml_path else TrainConfig()

def get_celltype_topic_activation(rna_anndata, groupby_key=["celltype"], aggregation="mean"):
    
    latent = rna_anndata.obsm["X_scdori"]  # shape (n_cells, num_topics)
    df_latent = pd.DataFrame(latent, columns=[f"Topic_{i}" for i in range(latent.shape[1])])
    df_latent[groupby_key] = rna_anndata.obs[groupby_key].values

    if aggregation == "median":
        df_grouped = df_latent.groupby(groupby_key).median()
    elif aggregation == "mean":
        df_grouped = df_latent.groupby(groupby_key).mean()
    else:
        raise ValueError("aggregation type is not known")

    return df_grouped.T

def get_rep_or_act_celltype_df(cell_tf, tf_names, rna_metacell, celltype_column_key, out_dir, label:Literal["act", "rep"]):
    # aggregating activity per celltype
    df_celltype_tf = pd.DataFrame(cell_tf, columns=tf_names)
    df_celltype_tf[celltype_column_key] = rna_metacell.obs[celltype_column_key].values
    df_celltype_tf = df_celltype_tf.groupby(celltype_column_key).mean()
    df_celltype_tf = df_celltype_tf.fillna(0)
    # removing TF with 0/Nan activity
    df_celltype_tf = df_celltype_tf.loc[:, (df_celltype_tf != 0).any(axis=0)]

    #### Plot top TF activity per celltype ####
    # top TFs per celltype
    tf_list_plot = []
    celltype_tf_marker = {}
    
    ## gathers celltype specific top 25 TF --> later benchmarking reseaons
    for k in df_celltype_tf.index:
        sorted_values = df_celltype_tf.loc[k].sort_values(ascending=False)[:25]
        celltype_tf_marker.update({k:list(set(sorted_values.index.values))})
        tf_list_plot = tf_list_plot +list(sorted_values[:5].index.values)
    tf_list_plot=list(set(tf_list_plot))

    plot_heatmap(
        df_plot=df_celltype_tf.T.loc[tf_list_plot,:].T,
        label=f"Top5_TF_celltype_{label}",
        outdir=out_dir
    )
    print(celltype_tf_marker)

    df_celltype_tf.to_csv(out_dir/ f"celltype_{label}_TF_activity.tsv", sep="\t")
    save_json(celltype_tf_marker, out_dir / f"{label}_celltype_TFs.json")

    return df_celltype_tf, celltype_tf_marker


def plot_umap(anndata_obj, colour_col:list|str, outdir):
    # visualing cell-types on scDoRI computed UMAP
    import seaborn as sns
    import scanpy as sc
    import matplotlib.pyplot as plt

    out_string =os.path.join(outdir, f'umap_{"_".join(colour_col) if isinstance(colour_col, list) else colour_col}.png')

    sns.set(font_scale=0.3)

    sns.set_style("whitegrid")
    with plt.rc_context({"figure.figsize": (8, 12), "figure.dpi": (600)}):
        umap_fig = sc.pl.umap(
            anndata_obj,
            color=colour_col,
            add_outline=True,
            outline_color=("white", "black"),
            size=10,
            #save=out_string,
            show=False,
            return_fig=True
        )

        umap_fig.savefig(out_string)
        plt.close(umap_fig)

def plot_heatmap(df_plot, label, outdir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(18, 6))  # Adjust width as needed
    
    # Create the heatmap
    g = sns.heatmap(
        df_plot,
        cmap='coolwarm',
        annot_kws={"size": 10, "square": True},
        center=0,
        square=True,
        linewidths=0.6,
        linecolor='white',
        cbar_kws={
            'label': label,
            'shrink': 0.9,  # Make colorbar slightly smaller
            'orientation': 'horizontal',   # 'vertical' or 'horizontal'
            'location': 'top',        # 'left', 'right', 'top', 'bottom'
            'aspect': 15,   # Make colorbar thinner
            'pad': 0.02,
            'format': '%.1f',
        },
        ax=ax
    )

    # Set x-axis labels explicitly to show all of them
    ax.set_xticks(range(len(df_plot.columns)))
    ax.set_xticklabels(df_plot.columns, rotation=90, ha='center', fontsize=8)
    # Add tick lines for x-axis
    ax.tick_params(axis='x', which='major', length=4, width=0.5, direction='out', bottom=True)
    
    # Set y-axis labels
    ax.set_yticks(range(len(df_plot.index)))
    ax.set_yticklabels(df_plot.index, rotation=0, fontsize=10)

    # Increase and rotate x-axis tick labels
    #g.tick_params(axis='x', labelsize=8, rotation=90)
    #g.tick_params(axis='y', labelsize=8, rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    fig.savefig(os.path.join(outdir, f"{label}.png"), dpi=800, bbox_inches='tight')
    plt.close()

def get_gene_peak_df(peak_celltype_df, gene_name:str, rna_metacell, atac_metacell, gene_peak, threshold=0.95):
    import numpy as np

    gene_index = list(rna_metacell.var_names).index(gene_name)
    enhancers = np.where(gene_peak[gene_index, :] > threshold)[0]
    enhancers = atac_metacell.var_names[enhancers]
    return peak_celltype_df.loc[enhancers]

def main(args):
    import logging
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from pathlib import Path
    from sklearn.preprocessing import OneHotEncoder
    import scanpy as sc
    

    from scdori import (
        load_scdori_inputs,
        set_seed,
        scDoRI,
        load_best_model,
        compute_neighbors_umap,
        compute_topic_peak_umap,
        compute_topic_gene_matrix,
        compute_atac_grn_activator_with_significance,
        compute_atac_grn_repressor_with_significance,
        compute_significant_grn,
        plot_topic_activation_heatmap,
        get_top_activators_per_topic,
        get_top_repressor_per_topic,
        compute_activator_tf_activity_per_cell,
        compute_repressor_tf_activity_per_cell,
        save_regulons,
        get_latent_topics,
        get_tf_expression
    )

    import os
    import gc
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    trainConfig = get_training_config(args.config_yaml)
    celltype_column_key = trainConfig.celltype_col
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=trainConfig.logging_level)

    ## set output directory ##
    data_dir = Path(trainConfig.data_dir)
    out_dir = data_dir / trainConfig.output_subdir / "postprocess"

    if not out_dir.exists():
        logger.info(f"Creating directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

    

    logger.info("Starting scDoRI downstream analysis")
    set_seed(trainConfig.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    rna_metacell, atac_metacell, gene_peak_dist, insilico_act, insilico_rep = (
        load_scdori_inputs(trainConfig)
    )   

    gene_peak_fixed = gene_peak_dist.clone()
    gene_peak_fixed[gene_peak_fixed > 0] = 1  # mask for peak-gene links based on distance

    logger.info(f"RNA adata:\n{rna_metacell}")
    logger.info(f"ATAC adata:\n{atac_metacell}")


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


    ## Building the scDoRI model
    num_genes = rna_metacell.n_vars
    num_peaks = atac_metacell.n_vars

    num_tfs = insilico_act.shape[1]

    num_batches = onehot_batch.shape[1]

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

    ##### Downstream Analysis ####
    model = load_best_model(
    model, Path(trainConfig.weights_folder_grn) / "best_scdori_best_eval.pth", device
    )

   # creating dataloader for all cells
    n_cells = rna_metacell.n_obs
    indices = np.arange(n_cells)

    all_dataset = TensorDataset(torch.from_numpy(indices))
    all_dataset_loader = DataLoader(
        all_dataset, batch_size=trainConfig.batch_size_cell_prediction, shuffle=False
    )

    # get scDoRI latent embedding (topics)
    scdori_latent = get_latent_topics(
        model,
        device,
        all_dataset_loader,
        rna_metacell,
        atac_metacell,
        num_cells,
        tf_indices,
        onehot_batch,
    )

    # adding scDoRI embedding to the anndata object
    rna_metacell.obsm["X_scdori"] = scdori_latent

    # computing neighbourhood graph and UMAP based on scDoRI embedding, UMAP parameters can be set in config file
    compute_neighbors_umap(
        rna_metacell,
        rep_key="X_scdori",
        umap_n_neighbors=trainConfig.umap_n_neighbors,
        umap_min_dist=trainConfig.umap_min_dist,
        umap_random_state=trainConfig.umap_random_state
    )

    plot_umap(
        anndata_obj=rna_metacell,
        colour_col=[celltype_column_key, trainConfig.batch_col],
        outdir=out_dir
    )

    
    df_topic_celltype = get_celltype_topic_activation(
    rna_metacell, groupby_key=[celltype_column_key], aggregation="mean"
    )

    ### remove not active topics ###
    # removing topics not active highly in any of the celltypes
    select_topics = [
        "Topic_" + str(k) for k in np.where(df_topic_celltype.max(axis=1) > 0.07)[0]
    ]

    plot_heatmap(
        df_plot=df_topic_celltype.loc[select_topics],
        label="Average_Topic_Activation",
        outdir=out_dir
    )

    ### Compute Top genes per topic ###
    topic_gene_embedding = compute_topic_gene_matrix(model, device)
    adata_gene = sc.AnnData(topic_gene_embedding)
    adata_gene.var.index = ["Topic_" + str(i) for i in range(model.num_topics)]
    adata_gene.obs.index = rna_metacell.var.index


    ### Compute peaks per topic ###
    umap_embedding_peaks, topic_peak_embedding = compute_topic_peak_umap(
        model,
        device,
        umap_n_neighbors=trainConfig.umap_n_neighbors,
        umap_min_dist=trainConfig.umap_min_dist,
        umap_random_state=trainConfig.umap_random_state
    )
    ## creating anndata with observations as peaks and values as topic association of each peak
    adata_peak = sc.AnnData(topic_peak_embedding)
    adata_peak.var.index = ["Topic_" + str(i) for i in range(model.num_topics)]
    adata_peak.obs.index = atac_metacell.var.index
    adata_peak.obsm["X_umap"] = umap_embedding_peaks

    atac_metacell.obs[celltype_column_key] = rna_metacell.obs[celltype_column_key].copy()

    # computing average accesiblity of peaks in each celltype
    atac_metacell.layers["counts"] = atac_metacell.X
    sc.pp.normalize_total(atac_metacell)
    aggregated_atac = sc.get.aggregate(atac_metacell, by=celltype_column_key, func=["mean"])
    aggregated_atac.X = aggregated_atac.layers["mean"]
    sc.pp.normalize_total(aggregated_atac)
    sc.pp.scale(aggregated_atac)

    # adding average accesibility of each peak in a celltype to peak anndata
    peak_celltype_df = aggregated_atac.to_df().T
    peak_celltype_df = peak_celltype_df.loc[adata_peak.obs.index.values]
    adata_peak.obs = pd.concat([adata_peak.obs, peak_celltype_df], axis=1)

    # adding insilico chipseq embeddings to peak anndata
    tf_names = rna_metacell.var[rna_metacell.var.gene_type == "TF"].index.values

    # Create a dictionary for activator and repressor insilico-chipseq binding score
    tf_binding_data = {
        tf_name + "_activator_binding": insilico_act[:, i].numpy()
        for i, tf_name in enumerate(tf_names)
    }
    tf_binding_data.update(
        {
            tf_name + "_repressor_binding": np.abs(insilico_rep[:, i].numpy())
            for i, tf_name in enumerate(tf_names)
        }
    )

    # Convert the dictionary to a DataFrame
    tf_binding_data_df = pd.DataFrame(tf_binding_data, index=adata_peak.obs.index)
    # Concatenate new columns with existing obs
    adata_peak.obs = pd.concat([adata_peak.obs, tf_binding_data_df], axis=1)

    ### Computing ATAC based GRNs with emprirical significance ###
    """
    these GRNs do not use evidence of TF-gene co-expression

    activator GRNs here indicate if within a topic, peaks linked to a gene have accesible binding sites for a TF (from activator insilico-chipseq scores)

    repressor GRNs here indicate if within a topic, peaks linked to a gene have non-accesible repressor binding sites for a TF (from repressor insilico-chipseq scores)

    additionally we compute a background set of GRN values by shuffling insilico-chipseq scores, which are used to compute empirical significance
    """
    if not os.path.exists(out_dir / "grn_act_atac"/ "grn_atac_activator_0.05.npy"):
        compute_atac_grn_activator_with_significance(
            model, device, cutoff_val=0.05, outdir=out_dir / "grn_act_atac", num_permutations=trainConfig.num_permutations
        )

    # ATAC based GRN for repressors
    if not os.path.exists(out_dir / "grn_act_atac"/ "grn_atac_repressor_0.05.npy"):
        compute_atac_grn_repressor_with_significance(
            model, device, cutoff_val=0.05, outdir=out_dir /"grn_act_atac", num_permutations=trainConfig.num_permutations
        )

    ### Compuy final GRNs ###
    # calculating TF-expression per topic
    # either from scdori model weights or from true data
    # using true expression here
    tf_normalised = get_tf_expression(
        #"True",
        trainConfig.tf_expression_mode,
        model,
        device,
        all_dataset_loader,
        rna_metacell,
        atac_metacell,
        num_cells,
        tf_indices,
        onehot_batch,
        trainConfig,
    )

    # compute final GRNs which use the significant ATAC based GRNs derived above
    grn_act, grn_rep = compute_significant_grn(
        model,
        device,
        cutoff_val_activator=0.05,
        cutoff_val_repressor=0.05,
        tf_normalised=tf_normalised.detach().cpu().numpy(),
        outdir= out_dir / "grn_act_atac",
    )

    if np.isnan(grn_act).sum() or np.isinf(grn_act).sum():
        # Check and clean data
        print("Checking for non-finite values in grn_act:")
        print(f"NaN count: {np.isnan(grn_act).sum()}")
        print(f"Inf count: {np.isinf(grn_act).sum()}")
        ##Clean the data
        grn_act_clean = np.nan_to_num(grn_act, nan=0.0, posinf=1e10, neginf=-1e10)
        grn_act = grn_act_clean.copy()

    if np.isnan(grn_rep).sum() or np.isinf(grn_rep).sum():
        # Check and clean data
        print("Checking for non-finite values in grn_rep:")
        print(f"NaN count: {np.isnan(grn_rep).sum()}")
        print(f"Inf count: {np.isinf(grn_rep).sum()}")
        ##Clean the data
        grn_rep_clean = np.nan_to_num(grn_rep, nan=0.0, posinf=1e10, neginf=-1e10)
        grn_rep = grn_rep_clean.copy()

    # save regulons per TF
    save_regulons(
        grn_act,
        tf_names=tf_names,
        gene_names=rna_metacell.var.index.values,
        num_topics=model.num_topics,
        output_dir=out_dir /"grn_act_atac",
        mode="activator",
    )

    # save regulons per TF
    save_regulons(
        grn_rep,
        tf_names=tf_names,
        gene_names=rna_metacell.var.index.values,
        num_topics=model.num_topics,
        output_dir=out_dir / "grn_act_atac",
        mode="repressor",
    )

    ## compute and plot top activator per topic ##
    # plotting TF activity across topics
    tf_names = rna_metacell.var[rna_metacell.var.gene_type == "TF"].index.values

    # plot top k activators per topic
    df_topic_activator, top_regulators = get_top_activators_per_topic(
        grn_act,
        tf_names,
        scdori_latent,
        selected_topics=None,
        top_k=5,
        clamp_value=1e-8,
        zscore=True,
        figsize=(25, 10),
        out_fig=out_dir / "top5_regulators.png",
    )
    plt.close('all')

    ## Top Tfs per topic
    selected_tf = set()
    for _i, row_name in enumerate(df_topic_activator.index):
        row = df_topic_activator.loc[row_name].sort_values(ascending=False)
        top_tfs = row.head(5).index.values
        selected_tf.update(top_tfs)
    selected_tf = list(selected_tf)
    selected_tf = sorted(selected_tf)

    df_plot = df_topic_activator[selected_tf]

    plot_heatmap(
        df_plot=df_plot,
        label="Top5_TF_topic_activator",
        outdir=out_dir
    )

    df_topic_activator.to_csv(out_dir/ "topic_TF_activity.tsv", sep="\t")
    

    scdori_latent_copy = scdori_latent.copy()
    scdori_latent_copy[scdori_latent_copy<0.1]=0

    # computing TF activity per cell
    cell_tf_act = compute_activator_tf_activity_per_cell(
        grn_act,
        tf_names,
        scdori_latent_copy,
        selected_topics=None,
        clamp_value=1e-8,
        zscore=True,
    )

    get_rep_or_act_celltype_df(
        cell_tf=cell_tf_act,
        tf_names=tf_names,
        rna_metacell=rna_metacell,
        celltype_column_key=celltype_column_key,
        out_dir=out_dir,
        label="act"
    )

    cell_tf_rep = compute_repressor_tf_activity_per_cell(
    grn_rep,
    tf_names,
    scdori_latent,
    selected_topics=None,
    clamp_value=1e-8,
    zscore=True,
    )

    get_rep_or_act_celltype_df(
        cell_tf=cell_tf_rep,
        tf_names=tf_names,
        rna_metacell=rna_metacell,
        celltype_column_key=celltype_column_key,
        out_dir=out_dir,
        label="rep"
    )

    ### Get enhancer gene links ###

    # peaks gene links used by scdori
    gene_peak = (model.gene_peak_factor_learnt.detach().cpu().numpy()) * (
        model.gene_peak_factor_fixed.detach().cpu().numpy()
    )


    ### calculate and save gene_peak interactions per celltype
    gene_peak_outdir = os.path.join(out_dir, "gene_peak_interactions")
    os.makedirs(gene_peak_outdir, exist_ok=True)

    for gene_name in rna_metacell.var.index:
        peak_gene_assoc_df = get_gene_peak_df(peak_celltype_df, gene_name, rna_metacell, atac_metacell, gene_peak, threshold=0.95)
        if peak_gene_assoc_df.empty: continue    
        peak_gene_assoc_df.to_csv(os.path.join(gene_peak_outdir, f"{gene_name}_enhancers_peak_links_.tsv"), sep="\t")

    ### Compute and save celltype specific GRN (TF-gene) and eGRNs (peak-gene) ####
    grn_outpath_dir = out_dir / "GRN_act_TF_gene"
    os.makedirs(grn_outpath_dir, exist_ok=True)

    GRN_act = get_celltype_GRN(
            grn_topic=grn_act,
            df_topic_celltype=df_topic_celltype,
            tf_names=tf_names,
            gene_names=rna_metacell.var.index,
            outdir=grn_outpath_dir
    )

    egrn_outpath_dir = out_dir / "eGRN_act_peak_gene"
    os.makedirs(egrn_outpath_dir, exist_ok=True)

    get_celltype_eGRN(
            GRN_celltype=GRN_act,
            gene_peak_tensor=gene_peak,
            df_topic_celltype=df_topic_celltype,
            peak_names=atac_metacell.var.index,
            tf_names=tf_names,
            outdir=egrn_outpath_dir

    )

    grn_outpath_dir = out_dir / "GRN_rep_TF_gene"
    os.makedirs(grn_outpath_dir, exist_ok=True)

    GRN_rep = get_celltype_GRN(
            grn_topic=grn_rep,
            df_topic_celltype=df_topic_celltype,
            tf_names=tf_names,
            gene_names=rna_metacell.var.index,
            outdir=grn_outpath_dir
    )

    egrn_outpath_dir = out_dir / "eGRN_rep_peak_gene"
    os.makedirs(egrn_outpath_dir, exist_ok=True)

    get_celltype_eGRN(
            GRN_celltype=GRN_rep,
            gene_peak_tensor=gene_peak,
            df_topic_celltype=df_topic_celltype,
            peak_names=atac_metacell.var.index,
            tf_names=tf_names,
            outdir=egrn_outpath_dir

    )

    ### Save data ###

    rna_metacell.write_h5ad(os.path.join(out_dir, "rna_metadata.h5ad"))
    atac_metacell.write_h5ad(os.path.join(out_dir, "atac_metadata.h5ad"))
    adata_peak.write_h5ad(os.path.join(out_dir, "adata_peak.h5ad"))
    adata_gene.write_h5ad(os.path.join(out_dir, "adata_gene.h5ad"))


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