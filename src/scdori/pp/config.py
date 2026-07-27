import logging
import yaml
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """
    Global configuration module for the single-cell multiome data pre-processing used in scDoRI.

    This module holds constants and parameters used across the pipeline, including:
    1. Logging settings
    2. Paths for data, genome references, and motif databases
    3. Species and genome assembly details
    4. HVG/peak selection parameters
    5. Promoter logic
    6. In-silico ChIP-seq correlation settings
    7. Other hyperparameters or default values

    Variables
    ---------
    random_seed : int
        Random seed for reproducibility.
    data_dir : str
        Directory where RNA and ATAC AnnData files are stored.
    genome_dir : str
        Directory containing genome FASTA and chromosome size files.
    motif_directory : str
        Directory containing .meme motif files for TF databases.
    output_subdir_name : str
        Name of the subdirectory within data_dir to store intermediate outputs.
    rna_adata_file_name : str
        Filename for the RNA AnnData file (H5AD).
    atac_adata_file_name : str
        Filename for the ATAC AnnData file (H5AD).
    species : str
        Species name, e.g., "mouse" or "human".
    genome_assembly : str
        Genome assembly version, e.g., "mm10" or "hg38".
    gtf_url : str or None
        URL to the GTF file override; if None, defaults are used based on species and assembly.
    chrom_sizes_url : str or None
        URL to the chromosome sizes file override; if None, defaults are used if known.
    fasta_url : str or None
        URL to the FASTA genome file override; if None, defaults are used if known.
    chrom_sizes_file : str
        Local filename for the chromosome sizes, e.g., "mm10.chrom.sizes" or "hg38.chrom.sizes".
    mitochondrial_prefix : str
        Prefix used to identify mitochondrial genes in the RNA AnnData.
    genes_user : list of str
        User-provided genes always included in the final model, even if not Highly variable (HV).
    tfs_user : list of str
        User-provided TFs always included in the final model, even if not HV.
    motif_database : str
        Name of the motif database, e.g., "cisbp".
    num_genes : int
        Number of genes to select for scDoRI training (via HVG + user overrides).
    num_tfs : int
        Number of TFs to select for scDoRI training (via HV among potential TFs + user overrides).
    min_cells_per_gene : int
        Minimum cell count threshold for gene detection (not enforced in current code).
    window_size : int
        Genomic window (bp) around each gene for selecting peaks. Example: 80,000 => ±80 kb.
    num_peaks : int
        Target number of peaks for training scDoRI; some are forced (promoters), the rest are HV.
    peak_std_batch_key : str
        Key in `.obs` used to group cells and measure peak standard deviation for HV selection.
    batch_key : str
        Key in `.obs` denoting experimental batch/covariate for integration or Harmony correction.
    leiden_resolution : float
        Resolution parameter for the Leiden clustering used to create metacells.
    keep_promoter_peaks : bool
        Whether to keep promoter peaks unconditionally in the final set of peaks.
    promoter_col : str
        Column in ATAC `.var` indicating if a peak is a promoter peak (True/False).
    motif_match_pvalue_threshold : float
        P-value threshold for motif hits in FIMO for in-silico ChIP-seq.
    correlation_percentile : float
        Percentile cutoff for correlation significance with TF expression (e.g., 99 => p≈0.01).
    n_bg_peaks_for_corr : int
        Number of peaks (lowest motif scores) used per TF as background in correlation tests.
    peak_distance_scaling_factor : float
        Decay factor for exponential distance weighting in initial gene-peak links.
    peak_distance_min_cutoff : float
        Minimum allowed scaled distance (exponential) threshold within the user-defined window.
    """
    # Logging
    logging_level: int = logging.INFO

    # Seed
    random_seed: int = 42

    # Directory structure
    data_dir: str = "/fast/AG_Bunina/Berk/workdir/scDoRI/exp_sarah"
    genome_dir: str = "/fast/AG_Bunina/Berk/workdir/scDoRI/genome"
    motif_directory: str = "/fast/AG_Bunina/Berk/projects/scDoRI/assets/motif_database"
    output_subdir_name: str = "outputs"

    # Input Filenames
    rna_adata_file_name: str = "merged_rna.h5ad"
    atac_adata_file_name: str = "merged_atac.h5ad"

    # Species & references
    species: str = "human"
    genome_assembly: str = "hg38"

    # Optional user-provided URLs for genome files
    gtf_url: Optional[str] = None
    chrom_sizes_url: Optional[str] = None
    fasta_url: Optional[str] = None

    chrom_sizes_file: str = "hg38.chrom.sizes"

    # Genes & TF selection
    mitochondrial_prefix: str = "mt-"
    genes_user: List[str] = None
    tfs_user: List[str] = None
    motif_database: str = "cisbp"

    num_genes: int = 4000
    num_tfs: int = 300
    min_cells_per_gene: int = 4
    ##mandatory genes
    gene_set_to_keep: str= ""

    # Genomic window
    window_size: int = 80000

    num_peaks: int = 100000
    peak_std_batch_key: str = "leiden"

    # Batch key & metacell parameters
    batch_key: str = "batch"
    leiden_resolution: float = 10

    # Promoter logic
    keep_promoter_peaks: bool = True
    promoter_col: str = "is_promoter"
    promoters_bed_file: str = ""

    # Mandatory peaks
    peak_set_to_keep: str = ""

    # Correlation & in-silico ChIP-seq
    motif_match_pvalue_threshold: float = 1e-3
    correlation_percentile: float = 99
    n_bg_peaks_for_corr: int = 5000

    # Distance matrix parameters
    peak_distance_scaling_factor: float = 20000

    def __post_init__(self):
        """Initialize default lists if None."""
        if self.genes_user is None:
            self.genes_user = []
        if self.tfs_user is None:
            self.tfs_user = []

    @property
    def peak_distance_min_cutoff(self) -> float:
        """Calculate the min cutoff based on window_size and scaling factor."""
        return np.e ** (-1 * (self.window_size / self.peak_distance_scaling_factor))

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PreprocessingConfig':
        """
        Load configuration from a YAML file.
        
        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.
            
        Returns
        -------
        PreprocessingConfig
            Configuration object with values from YAML file.
        """
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Map YAML structure to dataclass fields
        kwargs = {}
        
        # Logging
        if 'logging' in config_dict:
            log_level = config_dict['logging']['level']
            if isinstance(log_level, str):
                kwargs['logging_level'] = getattr(logging, log_level.upper())
            else:
                kwargs['logging_level'] = log_level
        
        # Random seed
        if 'random_seed' in config_dict:
            kwargs['random_seed'] = config_dict['random_seed']
        
        # Directories
        if 'directories' in config_dict:
            dir_config = config_dict['directories']
            kwargs.update({
                'data_dir': dir_config.get('data_dir'),
                'genome_dir': dir_config.get('genome_dir'),
                'motif_directory': dir_config.get('motif_directory'),
                'output_subdir_name': dir_config.get('output_subdir_name')
            })
        
        # Input files
        if 'input_files' in config_dict:
            file_config = config_dict['input_files']
            kwargs.update({
                'rna_adata_file_name': file_config.get('rna_adata_file_name'),
                'atac_adata_file_name': file_config.get('atac_adata_file_name')
            })
        
        # Genome
        if 'genome' in config_dict:
            genome_config = config_dict['genome']
            kwargs.update({
                'species': genome_config.get('species'),
                'genome_assembly': genome_config.get('assembly'),
                'gtf_url': genome_config.get('gtf_url'),
                'chrom_sizes_url': genome_config.get('chrom_sizes_url'),
                'fasta_url': genome_config.get('fasta_url'),
                'chrom_sizes_file': genome_config.get('chrom_sizes_file')
            })
        
        # Gene selection
        if 'gene_selection' in config_dict:
            gene_config = config_dict['gene_selection']
            kwargs.update({
                'mitochondrial_prefix': gene_config.get('mitochondrial_prefix'),
                'genes_user': gene_config.get('genes_user', []),
                'tfs_user': gene_config.get('tfs_user', []),
                'motif_database': gene_config.get('motif_database'),
                'num_genes': gene_config.get('num_genes'),
                'num_tfs': gene_config.get('num_tfs'),
                'min_cells_per_gene': gene_config.get('min_cells_per_gene'),
                'gene_set_to_keep': gene_config.get('gene_set_to_keep')
            })
        
        # Peak selection
        if 'peak_selection' in config_dict:
            peak_config = config_dict['peak_selection']
            kwargs.update({
                'window_size': peak_config.get('window_size'),
                'num_peaks': peak_config.get('num_peaks'),
                'peak_std_batch_key': peak_config.get('peak_std_batch_key'),
                'keep_promoter_peaks': peak_config.get('keep_promoter_peaks'),
                'promoter_col': peak_config.get('promoter_col'),
                'promoters_bed_file': peak_config.get('promoters_bed_file'),
                'peak_set_to_keep' : peak_config.get('peak_set_to_keep')
            })
        
        # Batch correction
        if 'batch_correction' in config_dict:
            batch_config = config_dict['batch_correction']
            kwargs.update({
                'batch_key': batch_config.get('batch_key'),
                'leiden_resolution': batch_config.get('leiden_resolution')
            })
        
        # Correlation
        if 'correlation' in config_dict:
            corr_config = config_dict['correlation']
            kwargs.update({
                'motif_match_pvalue_threshold': corr_config.get('motif_match_pvalue_threshold'),
                'correlation_percentile': corr_config.get('correlation_percentile'),
                'n_bg_peaks_for_corr': corr_config.get('n_bg_peaks_for_corr')
            })
        
        # Distance
        if 'distance' in config_dict:
            dist_config = config_dict['distance']
            kwargs.update({
                'peak_distance_scaling_factor': dist_config.get('peak_distance_scaling_factor')
            })
        
        # Remove None values to use defaults
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return cls(**kwargs)

    def save_yaml(self, yaml_path: str):
        """
        Save current configuration to a YAML file.
        
        Parameters
        ----------
        yaml_path : str
            Path where to save the YAML configuration file.
        """
        config_dict = {
            'logging': {
                'level': logging.getLevelName(self.logging_level)
            },
            'random_seed': self.random_seed,
            'directories': {
                'data_dir': self.data_dir,
                'genome_dir': self.genome_dir,
                'motif_directory': self.motif_directory,
                'output_subdir_name': self.output_subdir_name
            },
            'input_files': {
                'rna_adata_file_name': self.rna_adata_file_name,
                'atac_adata_file_name': self.atac_adata_file_name
            },
            'genome': {
                'species': self.species,
                'assembly': self.genome_assembly,
                'gtf_url': self.gtf_url,
                'chrom_sizes_url': self.chrom_sizes_url,
                'fasta_url': self.fasta_url,
                'chrom_sizes_file': self.chrom_sizes_file
            },
            'gene_selection': {
                'mitochondrial_prefix': self.mitochondrial_prefix,
                'genes_user': self.genes_user,
                'tfs_user': self.tfs_user,
                'motif_database': self.motif_database,
                'num_genes': self.num_genes,
                'num_tfs': self.num_tfs,
                'min_cells_per_gene': self.min_cells_per_gene,
                'gene_set_to_keep': self.gene_set_to_keep
            },
            'peak_selection': {
                'window_size': self.window_size,
                'num_peaks': self.num_peaks,
                'peak_std_batch_key': self.peak_std_batch_key,
                'keep_promoter_peaks': self.keep_promoter_peaks,
                'promoter_col': self.promoter_col,
                'peak_set_to_keep': self.peak_set_to_keep
            },
            'batch_correction': {
                'batch_key': self.batch_key,
                'leiden_resolution': self.leiden_resolution
            },
            'correlation': {
                'motif_match_pvalue_threshold': self.motif_match_pvalue_threshold,
                'correlation_percentile': self.correlation_percentile,
                'n_bg_peaks_for_corr': self.n_bg_peaks_for_corr
            },
            'distance': {
                'peak_distance_scaling_factor': self.peak_distance_scaling_factor
            }
        }
        
        with open(yaml_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)


# Usage examples:

# Load from YAML file
# config = PreprocessingConfig.from_yaml('config.yaml')

# Create with defaults and save to YAML
# default_config = PreprocessingConfig()
# default_config.save_yaml('default_config.yaml')

# Use in your application
# ppConfig = PreprocessingConfig.from_yaml('path/to/your/config.yaml')