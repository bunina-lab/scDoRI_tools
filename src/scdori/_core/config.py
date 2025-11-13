import logging
import yaml
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TrainConfig:
    """
    Global configuration for the scDoRI modeling pipeline.
    """
    # LOGGING
    logging_level: int = logging.INFO

    # DATA PATHS
    data_dir: Path = Path("./")
    output_subdir: str = "outputs"
    rna_metacell_file: str = "rna_processed.h5ad"
    atac_metacell_file: str = "atac_processed.h5ad"
    batch_col: str = "batch"
    celltype_col:str = "celltype"
    gene_peak_distance_file: str = "gene_peak_distance_exp.npy"
    insilico_chipseq_act_file: str = "insilico_chipseq_act.npy"
    insilico_chipseq_rep_file: str = "insilico_chipseq_rep.npy"

    # RANDOM SEED
    random_seed: int = 42

    # BATCH / ARCHITECTURE
    batch_size_cell: int = 128
    dim_encoder1: int = 500
    dim_encoder2: int = 200
    num_topics: int = 40
    batch_size_cell_prediction: int = 256

    # PHASE1
    epoch_warmup_1: int = 5
    max_scdori_epochs: int = 1000

    # PHASE 2
    max_grn_epochs: int = 1000
    update_encoder_in_grn: bool = False
    update_peak_gene_in_grn: bool = False
    update_topic_peak_in_grn: bool = False
    update_topic_tf_in_grn: bool = False

    # early stopping and evaluation
    eval_frequency: int = 1
    phase1_patience: int = 50
    grn_val_patience: int = 5

    # LR / LOSSES
    learning_rate_scdori: float = 0.005
    learning_rate_grn: float = 0.001

    # Phase 1 weights (warmup_1)
    weight_atac_phase1: float = 1.0
    weight_tf_phase1: float = 200.0
    weight_rna_phase1: float = 0.0
    weight_rna_grn_phase1: float = 0.0

    # Phase 1 weights (warmup_2)
    weight_atac_phase2: float = 1.0
    weight_tf_phase2: float = 200.0
    weight_rna_phase2: float = 20.0
    weight_rna_grn_phase2: float = 0.0

    # Phase 2 (GRN) weights
    weight_atac_grn: float = 1.0
    weight_tf_grn: float = 200.0
    weight_rna_grn: float = 20.0
    weight_rna_from_grn: float = 20.0

    # REGULARIZATION
    l1_penalty_topic_tf: float = 0.001
    l2_penalty_topic_tf: float = 0.000
    l1_penalty_topic_peak: float = 0.001
    l2_penalty_topic_peak: float = 0.001
    l1_penalty_gene_peak: float = 0.001
    l2_penalty_gene_peak: float = 0.005
    l1_penalty_grn_activator: float = 0.00005
    l1_penalty_grn_repressor: float = 0.0000

    # TF EXPRESSION SETTINGS
    tf_expression_mode: str = "True" ## True or latent # FIXME
    tf_expression_clamp: float = 0.1
    cells_per_topic: int = 200

    # SAVE FOLDERS (will be set in __post_init__)
    weights_folder_scdori: str = ""
    weights_folder_grn: str = ""
    best_scdori_model_path: str = ""
    best_grn_model_path: str = ""

    # UMAP PARAMETERS
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_random_state: int = 42

    # SIGNIFICANCE SETTINGS
    significance_cutoffs: List[float] = None
    num_permutations: int = 1000

    def __post_init__(self):
        """Initialize default lists and construct file paths."""
        if self.significance_cutoffs is None:
            self.significance_cutoffs = [0.001, 0.005, 0.01, 0.05]
        
        # Set default paths if not already set
        if not self.weights_folder_scdori:
            self.weights_folder_scdori = os.path.join(self.data_dir, "weights", "scdori")
        if not self.weights_folder_grn:
            self.weights_folder_grn = os.path.join(self.data_dir, "weights", "grn")
        if not self.best_scdori_model_path:
            self.best_scdori_model_path = os.path.join(self.weights_folder_grn, "best_scdori_final.pth")
        if not self.best_grn_model_path:
            self.best_grn_model_path = os.path.join(self.weights_folder_grn, "best_grn.pth")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainConfig':
        """
        Load configuration from a YAML file.
        
        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.
            
        Returns
        -------
        TrainConfig
            Configuration object with values from YAML file.
        """
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
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
        
        # Data paths
        if 'data' in config_dict:
            data_config = config_dict['data']
            if 'data_dir' in data_config:
                kwargs['data_dir'] = Path(data_config['data_dir'])
            kwargs.update({
                'output_subdir': data_config.get('output_subdir'),
                'rna_metacell_file': data_config.get('rna_metacell_file'),
                'atac_metacell_file': data_config.get('atac_metacell_file'),
                'batch_col': data_config.get('batch_col'),
                'gene_peak_distance_file': data_config.get('gene_peak_distance_file'),
                'insilico_chipseq_act_file': data_config.get('insilico_chipseq_act_file'),
                'insilico_chipseq_rep_file': data_config.get('insilico_chipseq_rep_file'),
                'celltype_col': data_config.get('celltype_col'),
            })
        
        # Architecture
        if 'architecture' in config_dict:
            arch_config = config_dict['architecture']
            kwargs.update({
                'batch_size_cell': arch_config.get('batch_size_cell'),
                'dim_encoder1': arch_config.get('dim_encoder1'),
                'dim_encoder2': arch_config.get('dim_encoder2'),
                'num_topics': arch_config.get('num_topics'),
                'batch_size_cell_prediction': arch_config.get('batch_size_cell_prediction')
            })
        
        # Training
        if 'training' in config_dict:
            train_config = config_dict['training']
            
            # Phase 1
            if 'phase1' in train_config:
                phase1_config = train_config['phase1']
                kwargs.update({
                    'epoch_warmup_1': phase1_config.get('epoch_warmup_1'),
                    'max_scdori_epochs': phase1_config.get('max_scdori_epochs')
                })
                
                # Warmup 1 weights
                if 'weights_warmup_1' in phase1_config:
                    w1 = phase1_config['weights_warmup_1']
                    kwargs.update({
                        'weight_atac_phase1': w1.get('weight_atac'),
                        'weight_tf_phase1': w1.get('weight_tf'),
                        'weight_rna_phase1': w1.get('weight_rna'),
                        'weight_rna_grn_phase1': w1.get('weight_rna_grn')
                    })
                
                # Warmup 2 weights
                if 'weights_warmup_2' in phase1_config:
                    w2 = phase1_config['weights_warmup_2']
                    kwargs.update({
                        'weight_atac_phase2': w2.get('weight_atac'),
                        'weight_tf_phase2': w2.get('weight_tf'),
                        'weight_rna_phase2': w2.get('weight_rna'),
                        'weight_rna_grn_phase2': w2.get('weight_rna_grn')
                    })
            
            # Phase 2
            if 'phase2' in train_config:
                phase2_config = train_config['phase2']
                kwargs.update({
                    'max_grn_epochs': phase2_config.get('max_grn_epochs')
                })
                
                # Update components
                if 'update_components' in phase2_config:
                    updates = phase2_config['update_components']
                    kwargs.update({
                        'update_encoder_in_grn': updates.get('update_encoder_in_grn'),
                        'update_peak_gene_in_grn': updates.get('update_peak_gene_in_grn'),
                        'update_topic_peak_in_grn': updates.get('update_topic_peak_in_grn'),
                        'update_topic_tf_in_grn': updates.get('update_topic_tf_in_grn')
                    })
                
                # GRN weights
                if 'weights_grn' in phase2_config:
                    grn_weights = phase2_config['weights_grn']
                    kwargs.update({
                        'weight_atac_grn': grn_weights.get('weight_atac'),
                        'weight_tf_grn': grn_weights.get('weight_tf'),
                        'weight_rna_grn': grn_weights.get('weight_rna'),
                        'weight_rna_from_grn': grn_weights.get('weight_rna_from_grn')
                    })
        
        # Evaluation
        if 'evaluation' in config_dict:
            eval_config = config_dict['evaluation']
            kwargs.update({
                'eval_frequency': eval_config.get('eval_frequency'),
                'phase1_patience': eval_config.get('phase1_patience'),
                'grn_val_patience': eval_config.get('grn_val_patience')
            })
        
        # Learning rates
        if 'learning_rates' in config_dict:
            lr_config = config_dict['learning_rates']
            kwargs.update({
                'learning_rate_scdori': lr_config.get('learning_rate_scdori'),
                'learning_rate_grn': lr_config.get('learning_rate_grn')
            })
        
        # Regularization
        if 'regularization' in config_dict:
            reg_config = config_dict['regularization']
            kwargs.update({
                'l1_penalty_topic_tf': reg_config.get('l1_penalty_topic_tf'),
                'l2_penalty_topic_tf': reg_config.get('l2_penalty_topic_tf'),
                'l1_penalty_topic_peak': reg_config.get('l1_penalty_topic_peak'),
                'l2_penalty_topic_peak': reg_config.get('l2_penalty_topic_peak'),
                'l1_penalty_gene_peak': reg_config.get('l1_penalty_gene_peak'),
                'l2_penalty_gene_peak': reg_config.get('l2_penalty_gene_peak'),
                'l1_penalty_grn_activator': reg_config.get('l1_penalty_grn_activator'),
                'l1_penalty_grn_repressor': reg_config.get('l1_penalty_grn_repressor')
            })
        
        # TF expression
        if 'tf_expression' in config_dict:
            tf_config = config_dict['tf_expression']
            kwargs.update({
                'tf_expression_mode': tf_config.get('mode'),
                'tf_expression_clamp': tf_config.get('clamp'),
                'cells_per_topic': tf_config.get('cells_per_topic')
            })
        
        # Model saving paths
        if 'model_saving' in config_dict:
            save_config = config_dict['model_saving']
            data_dir = kwargs.get('data_dir', Path("./"))
            kwargs.update({
                'weights_folder_scdori': os.path.join(data_dir, save_config.get('weights_folder_scdori', 'weights/scdori')),
                'weights_folder_grn': os.path.join(data_dir, save_config.get('weights_folder_grn', 'weights/grn')),
                'best_scdori_model_path': os.path.join(data_dir, save_config.get('best_scdori_model_path', 'weights/grn/best_scdori_final.pth')),
                'best_grn_model_path': os.path.join(data_dir, save_config.get('best_grn_model_path', 'weights/grn/best_grn.pth'))
            })
        
        # UMAP
        if 'umap' in config_dict:
            umap_config = config_dict['umap']
            kwargs.update({
                'umap_n_neighbors': umap_config.get('n_neighbors'),
                'umap_min_dist': umap_config.get('min_dist'),
                'umap_random_state': umap_config.get('random_state')
            })
        
        # Significance testing
        if 'significance_testing' in config_dict:
            sig_config = config_dict['significance_testing']
            kwargs.update({
                'significance_cutoffs': sig_config.get('cutoffs'),
                'num_permutations': sig_config.get('num_permutations')
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
        # Convert paths to relative paths for cleaner YAML
        data_dir_str = str(self.data_dir)
        
        config_dict = {
            'logging': {
                'level': logging.getLevelName(self.logging_level)
            },
            'random_seed': self.random_seed,
            'data': {
                'data_dir': data_dir_str,
                'output_subdir': self.output_subdir,
                'rna_metacell_file': self.rna_metacell_file,
                'atac_metacell_file': self.atac_metacell_file,
                'batch_col': self.batch_col,
                'celltype_col': self.celltype_col,
                'gene_peak_distance_file': self.gene_peak_distance_file,
                'insilico_chipseq_act_file': self.insilico_chipseq_act_file,
                'insilico_chipseq_rep_file': self.insilico_chipseq_rep_file
            },
            'architecture': {
                'batch_size_cell': self.batch_size_cell,
                'dim_encoder1': self.dim_encoder1,
                'dim_encoder2': self.dim_encoder2,
                'num_topics': self.num_topics,
                'batch_size_cell_prediction': self.batch_size_cell_prediction
            },
            'training': {
                'phase1': {
                    'epoch_warmup_1': self.epoch_warmup_1,
                    'max_scdori_epochs': self.max_scdori_epochs,
                    'weights_warmup_1': {
                        'weight_atac': self.weight_atac_phase1,
                        'weight_tf': self.weight_tf_phase1,
                        'weight_rna': self.weight_rna_phase1,
                        'weight_rna_grn': self.weight_rna_grn_phase1
                    },
                    'weights_warmup_2': {
                        'weight_atac': self.weight_atac_phase2,
                        'weight_tf': self.weight_tf_phase2,
                        'weight_rna': self.weight_rna_phase2,
                        'weight_rna_grn': self.weight_rna_grn_phase2
                    }
                },
                'phase2': {
                    'max_grn_epochs': self.max_grn_epochs,
                    'update_components': {
                        'update_encoder_in_grn': self.update_encoder_in_grn,
                        'update_peak_gene_in_grn': self.update_peak_gene_in_grn,
                        'update_topic_peak_in_grn': self.update_topic_peak_in_grn,
                        'update_topic_tf_in_grn': self.update_topic_tf_in_grn
                    },
                    'weights_grn': {
                        'weight_atac': self.weight_atac_grn,
                        'weight_tf': self.weight_tf_grn,
                        'weight_rna': self.weight_rna_grn,
                        'weight_rna_from_grn': self.weight_rna_from_grn
                    }
                }
            },
            'evaluation': {
                'eval_frequency': self.eval_frequency,
                'phase1_patience': self.phase1_patience,
                'grn_val_patience': self.grn_val_patience
            },
            'learning_rates': {
                'learning_rate_scdori': self.learning_rate_scdori,
                'learning_rate_grn': self.learning_rate_grn
            },
            'regularization': {
                'l1_penalty_topic_tf': self.l1_penalty_topic_tf,
                'l2_penalty_topic_tf': self.l2_penalty_topic_tf,
                'l1_penalty_topic_peak': self.l1_penalty_topic_peak,
                'l2_penalty_topic_peak': self.l2_penalty_topic_peak,
                'l1_penalty_gene_peak': self.l1_penalty_gene_peak,
                'l2_penalty_gene_peak': self.l2_penalty_gene_peak,
                'l1_penalty_grn_activator': self.l1_penalty_grn_activator,
                'l1_penalty_grn_repressor': self.l1_penalty_grn_repressor
            },
            'tf_expression': {
                'mode': self.tf_expression_mode,
                'clamp': self.tf_expression_clamp,
                'cells_per_topic': self.cells_per_topic
            },
            'model_saving': {
                'weights_folder_scdori': os.path.relpath(self.weights_folder_scdori, data_dir_str),
                'weights_folder_grn': os.path.relpath(self.weights_folder_grn, data_dir_str),
                'best_scdori_model_path': os.path.relpath(self.best_scdori_model_path, data_dir_str),
                'best_grn_model_path': os.path.relpath(self.best_grn_model_path, data_dir_str)
            },
            'umap': {
                'n_neighbors': self.umap_n_neighbors,
                'min_dist': self.umap_min_dist,
                'random_state': self.umap_random_state
            },
            'significance_testing': {
                'cutoffs': self.significance_cutoffs,
                'num_permutations': self.num_permutations
            }
        }
        
        with open(yaml_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)


# Usage examples:

# Load from YAML file
# trainConfig = TrainConfig.from_yaml('train_config.yaml')

# Create with defaults and save to YAML
# default_config = TrainConfig()
# default_config.save_yaml('default_train_config.yaml')