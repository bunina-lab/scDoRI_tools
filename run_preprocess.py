import logging
from pathlib import Path
import numpy as np
import pandas as pd

from scdori.pp import (
    PreprocessingConfig,
    compute_gene_peak_distance_matrix,
    compute_hvgs_and_tfs,
    compute_in_silico_chipseq,
    compute_motif_scores,
    create_dir_if_not_exists,
    create_extended_gene_bed,
    create_metacells,
    download_genome_references,
    filter_protein_coding_genes,
    intersect_cells,
    keep_promoters_and_select_hv_peaks,
    keep_promoter_mandatory_and_hv_peaks,
    load_anndata,
    load_gtf,
    load_motif_database,
    remove_mitochondrial_genes,
    run_bedtools_intersect,
    save_processed_datasets,
)


def get_pp_config(yaml_path=None):
    return PreprocessingConfig.from_yaml(yaml_path) if yaml_path else PreprocessingConfig()

def main(args):
    ppConfig = get_pp_config(args.config_yaml)



    logger = logging.getLogger(__name__)

    logging.getLogger().setLevel(ppConfig.logging_level)

    logger.info("=== Starting multi-ome preprocessing pipeline ===")

    data_dir = Path(ppConfig.data_dir)
    genome_dir = Path(ppConfig.genome_dir)
    motif_dir = Path(ppConfig.motif_directory)
    out_dir = data_dir / ppConfig.output_subdir_name

    create_dir_if_not_exists(genome_dir)
    create_dir_if_not_exists(motif_dir)
    create_dir_if_not_exists(out_dir)
    
    ### Load RNA and ATAC data
    data_rna, data_atac = load_anndata(
    data_dir, ppConfig.rna_adata_file_name, ppConfig.atac_adata_file_name
    )

    data_rna, data_atac = intersect_cells(data_rna, data_atac)

    data_rna = remove_mitochondrial_genes(
            data_rna, mito_prefix=ppConfig.mitochondrial_prefix
        )

    gtf_file = genome_dir / "annotation.gtf"
    gtf_df = load_gtf(gtf_file)
    data_rna = filter_protein_coding_genes(data_rna, gtf_df)

    motif_path = motif_dir / f"{ppConfig.motif_database}_{ppConfig.species}.meme"
    tf_names_all = []
    with open(motif_path) as f:
        for line in f:
            if line.startswith("MOTIF"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    tf_name = parts[2].split("_")[0].strip("()").strip()
                    tf_names_all.append(tf_name)
    tf_names_all = sorted(list(set(tf_names_all)))

    ### Gene Selection ##
    user_genes = ppConfig.genes_user
    user_tfs = ppConfig.tfs_user

    if ppConfig.gene_set_to_keep and Path(ppConfig.gene_set_to_keep).is_file():
        ## get genes and tfs from the txt file
        with open(ppConfig.gene_set_to_keep, "r") as fh:
            user_forced_genes = list({line.replace("\n", "").strip() for line in fh if line.strip()})
        
        for u_gene in user_forced_genes:
            if u_gene in tf_names_all:
                user_tfs.append(u_gene)
            else:
                user_genes.append(u_gene)

    data_rna, final_genes, final_tfs = compute_hvgs_and_tfs(
        data_rna=data_rna,
        tf_names=tf_names_all,
        user_genes=user_genes,
        user_tfs=user_tfs,
        num_genes=ppConfig.num_genes,
        num_tfs=ppConfig.num_tfs,
        min_cells=ppConfig.min_cells_per_gene,
    )

    chrom_sizes_path = genome_dir / f"{ppConfig.genome_assembly}.chrom.sizes"
    extended_genes_bed_df = create_extended_gene_bed(
        gtf_df,
        final_genes + final_tfs,  # if we want to include TF genes too
        window_size=ppConfig.window_size,
        chrom_sizes_path=chrom_sizes_path,
    )

    gene_bed_file = out_dir / f"genes_extended_{ppConfig.window_size//1000}kb.bed"
    extended_genes_bed_df.to_csv(gene_bed_file, sep="\t", header=False, index=False)
    logger.info(f"Created extended gene bed => {gene_bed_file}")


    ## Split and save the peak bed.
    if ":" in data_atac.var_names[0]: ##
        data_atac.var["chr"] = [v.split(":")[0] for v in data_atac.var_names]
        data_atac.var["start"] = [
            int(v.split(":")[1].split("-")[0]) if "-" in v else int(v.split(":")[1].split("_")[0]) for v in data_atac.var_names
        ]
        data_atac.var["end"] = [
            int(v.split(":")[1].split("-")[1]) if "-" in v else int(v.split(":")[1].split("_")[1]) for v in data_atac.var_names
            ]
    
    elif "_" in data_atac.var_names[0]:
        data_atac.var["chr"] = [v.split("_")[0] for v in data_atac.var_names]
        data_atac.var["start"] = [
            int(v.split("_")[1].split("-")[0]) if "-" in v else int(v.split("_")[1]) for v in data_atac.var_names
        ]
        data_atac.var["end"] = [
            int(v.split("_")[1].split("-")[1]) if "-" in v else int(v.split("_")[2]) for v in data_atac.var_names
            ]
    
    elif "-" in data_atac.var_names[0]:
        data_atac.var["chr"] = [v.split("-")[0] for v in data_atac.var_names]
        data_atac.var["start"] = [int(v.split("-")[1]) for v in data_atac.var_names]
        data_atac.var["end"] = [int(v.split("-")[2]) for v in data_atac.var_names]
    
    else:
        raise ValueError(f"Could not parse the peak column:\n{data_atac.var_names[0]}")
    
    data_atac.var["peak_name"] = data_atac.var.index
    all_peaks_bed = out_dir / "peaks_all.bed"
    data_atac.var[["chr", "start", "end", "peak_name"]].to_csv(
        all_peaks_bed, sep="\t", header=False, index=False
    )


    ### Intersect peak with gene window
    #
    intersected_bed = out_dir / "peaks_intersected.bed"
    run_bedtools_intersect(
        a_bed=all_peaks_bed, b_bed=gene_bed_file, out_bed=intersected_bed
    )

    peaks_intersected = pd.read_csv(intersected_bed, sep="\t", header=None)
    peaks_intersected.columns = ["chr", "start", "end", "peak_name"]
    windowed_set = set(peaks_intersected["peak_name"])

    # Subset data_atac to these peaks
    data_atac = data_atac[:, list(windowed_set)].copy()
    logger.info(f"After gene-window filtering => shape={data_atac.shape}")

    #
    rna_metacell, atac_metacell = create_metacells(
        data_rna,
        data_atac,
        grouping_key="leiden",
        resolution=ppConfig.leiden_resolution,
        batch_key=ppConfig.batch_key,
    )
    # Copy labels
    data_atac.obs["leiden"] = data_rna.obs["leiden"]


    ### Get promoter regions 
    if ppConfig.keep_promoter_peaks and ppConfig.promoters_bed_file:
        import pyranges as pr
        ##Read promoters file
        promoters_df = pd.read_csv(ppConfig.promoters_bed_file, sep="\t", header=None)
        promoters_df.columns = ["Chromosome", "Start", "End", "gene_name"]
        ###
      # Convert to PyRanges objects, preserving the original index
        atac_df = data_atac.var.copy()
        atac_df['original_index'] = data_atac.var.index  # Preserve original index

        atac_gr = pr.PyRanges(
            chromosomes=atac_df["chr"].values,
            starts=atac_df["start"].values,
            ends=atac_df["end"].values
        )

        promoters_gr = pr.PyRanges(
            promoters_df
        )

        # Find overlaps
        overlaps = atac_gr.overlap(promoters_gr)
        # Create boolean array based on which rows are in overlaps
        data_atac.var[ppConfig.promoter_col] = False
        if len(overlaps) > 0:
            # Get the integer positions of overlapping peaks
            overlap_positions = overlaps.df.index.values
            # Use iloc to set by position
            data_atac.var.iloc[overlap_positions, data_atac.var.columns.get_loc(ppConfig.promoter_col)] = True
    
    
    if ppConfig.peak_set_to_keep:
        with open(ppConfig.peak_set_to_keep, "r") as fh:
            peak_set_to_keep = set([line.strip().replace("\n", "") for line in fh])


    data_atac = keep_promoters_and_select_hv_peaks(
        data_atac=data_atac,
        total_n_peaks=ppConfig.num_peaks,
        cluster_key="leiden",
        promoter_col=ppConfig.promoter_col,  # column in data_atac.var
    ) if ppConfig.peak_set_to_keep is None else \
        keep_promoter_mandatory_and_hv_peaks(
            data_atac=data_atac,
            total_n_peaks=ppConfig.num_peaks,
            cluster_key="leiden",
            promoter_col=ppConfig.promoter_col,  # column in data_atac.var
            peak_set_to_keep=peak_set_to_keep
        )

    logger.info(f"Final shape after combining promoters + HV => {data_atac.shape}")

    save_processed_datasets(data_rna, data_atac, out_dir)

    if ":" in data_atac.var_names[0]: ##
        data_atac.var["chr"] = [v.split(":")[0] for v in data_atac.var_names]
        data_atac.var["start"] = [
            int(v.split(":")[1].split("-")[0]) if "-" in v else int(v.split(":")[1].split("_")[0]) for v in data_atac.var_names
        ]
        data_atac.var["end"] = [
            int(v.split(":")[1].split("-")[1]) if "-" in v else int(v.split(":")[1].split("_")[1]) for v in data_atac.var_names
            ]
    
    elif "_" in data_atac.var_names[0]:
        data_atac.var["chr"] = [v.split("_")[0] for v in data_atac.var_names]
        data_atac.var["start"] = [
            int(v.split("_")[1].split("-")[0]) if "-" in v else int(v.split("_")[1]) for v in data_atac.var_names
        ]
        data_atac.var["end"] = [
            int(v.split("_")[1].split("-")[1]) if "-" in v else int(v.split("_")[2]) for v in data_atac.var_names
            ]
    
    elif "-" in data_atac.var_names[0]:
        data_atac.var["chr"] = [v.split("-")[0] for v in data_atac.var_names]
        data_atac.var["start"] = [int(v.split("-")[1]) for v in data_atac.var_names]
        data_atac.var["end"] = [int(v.split("-")[2]) for v in data_atac.var_names]
    
    else:
        raise ValueError(f"Could not parse the peak column:\n{data_atac.var_names[0]}")
    
    
    data_atac.var["peak_name"] = data_atac.var_names
    peaks_bed = out_dir / "peaks_selected.bed"
    data_atac.var[["chr", "start", "end", "peak_name"]].to_csv(
        peaks_bed, sep="\t", header=False, index=False
    )

    ### Compute motif matches ###
    ### We use FIMO module from tangermeme (https://tangermeme.readthedocs.io/en/latest/tutorials/Tutorial_D1_FIMO.html) to score the motifs ####

    motif_path = (
        Path(ppConfig.motif_directory)
        / f"{ppConfig.motif_database}_{ppConfig.species}.meme"
    )
    pwms_sub, key_to_tf = load_motif_database(motif_path, final_tfs)
    fasta_path = genome_dir / f"{ppConfig.genome_assembly}.fa"
    df_motif_scores = compute_motif_scores(
        bed_file=peaks_bed,
        fasta_file=fasta_path,
        pwms_sub=pwms_sub,
        key_to_tf=key_to_tf,
        n_peaks=data_atac.shape[1],
        window=500,
        threshold=ppConfig.motif_match_pvalue_threshold,
    )
    df_motif_scores = df_motif_scores[final_tfs]

    ### Save motif scores ###
    df_motif_scores.to_csv(out_dir / "motif_scores.tsv", sep="\t")

    # 14) Recompute metacells for correlation with selected peaks
    #     Or subset existing atac_metacell to the new set of peaks
    # then compute insilico-chipseq
    atac_metacell = atac_metacell[:, data_atac.var_names].copy()
    tf_mask = rna_metacell.var["gene_type"] == "TF"
    rna_matrix = rna_metacell.X[:, tf_mask]  # shape=(n_meta, n_tfs)
    atac_matrix = atac_metacell.X  # shape=(n_meta, n_peaks)

    insilico_chipseq_act, insilico_chipseq_rep = compute_in_silico_chipseq(
        atac_matrix=atac_matrix,
        rna_matrix=rna_matrix,
        motif_scores=df_motif_scores,
        percentile=ppConfig.correlation_percentile,
        n_bg=ppConfig.n_bg_peaks_for_corr,
    )
    np.save(out_dir / "insilico_chipseq_act.npy", insilico_chipseq_act)
    np.save(out_dir / "insilico_chipseq_rep.npy", insilico_chipseq_rep)

    # distance is set to 0 if the peak midpoint is within gene-body or promoter (5kb upstream of TSS by default)
    # distance is -1 if peak-gene pairs on different chromosomes

    data_atac.var["index_int"] = range(data_atac.shape[1])
    selected_peak_indices = data_atac.var["index_int"].values

    # Subset GTF to final genes
    gene_info = gtf_df[gtf_df.feature == "gene"].drop_duplicates("gene_name")
    gene_info["gene"] = gene_info["gene_name"].values
    gene_info = gene_info.set_index("gene_name")
    gene_info = gene_info.loc[data_rna.var_names.intersection(gene_info.index)]

    gene_info["chr"] = gene_info["seqname"]  # rename col for consistency
    # Create gene_coordinates_intersect with necessary columns
    gene_info = gene_info[["chr", "start", "end", "strand", "gene"]].copy()
    gene_info.columns = ["chr_gene", "start", "end", "strand", "gene"]

    dist_matrix = compute_gene_peak_distance_matrix(
        data_rna=data_rna, data_atac=data_atac, gene_coordinates_intersect=gene_info
    )
    np.save(out_dir / "gene_peak_distance_raw.npy", dist_matrix)

    dist_matrix[dist_matrix < 0] = 1e8
    dist_matrix = np.exp(
        -1 * dist_matrix.astype(float) / ppConfig.peak_distance_scaling_factor
    )
    dist_matrix = np.where(dist_matrix < ppConfig.peak_distance_min_cutoff, 0, dist_matrix)
    np.save(out_dir / "gene_peak_distance_exp.npy", dist_matrix)

    logger.info("=== Pipeline completed successfully ===")

    ppConfig.save_yaml(Path(data_dir) / "preprocess_config.yaml")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-ome preprocessing pipeline.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to YAML configuration file. If not provided, default config is used.",
    )
    args = parser.parse_args()
    main(args)