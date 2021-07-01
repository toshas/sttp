# Configuring Experiments

## Creating project environment config

The training script allows to specify multiple `yml` config files, which will be concatenated during execution. 
This is done to separate experiment configs from environment configs. 
To start running experiments, create your own config file with a number of environment settings, similar to those 
`configs/env_*.yml`. 
The settings are as follows:

```yaml
root_datasets:
  # Path where CIFAR-10 will be downloaded upon the first run
  cifar10: /raid/obukhova/datasets/torchvision
  # Path where STL-10 will be downloaded upon the first run
  stl10_48: /raid/obukhova/datasets/torchvision
# Path for local logs, each stored a separate folder named after the config file
root_logs: /raid/obukhova/logs
# Path for local W&B logs
root_wandb: /raid/obukhova/logs_wandb
# Boolean: Whether datasets should be downloaded before using (downloads only once,
# but may annoy with unpacking)
dataset_download: True
# List: Environment variables to make sure are set
assert_env_set:
    - CUDA_VISIBLE_DEVICES
# int: Interval for loss logging to tensorboard and W&B
num_log_loss_steps: 10
# int: Number of top singular values to keep in visualizations
vis_truncate_singular_values: 32
# int: Number of worker threads working in parallel to load training samples
workers: 4
# int: Number of worker threads working in parallel to load validation samples
workers_validation: 4
# str: Name of the project in W&B
wandb_project: sttp
```

## GAN Configuration Template

The following config template should be used with `train_gan.py` training script:

```yaml
# sngan: Experiment type
experiment: sngan                
# str: __FILENAMEBASE__ will be substituted with the file name, any other 
# string will be used as is
experiment_name: __FILENAMEBASE__       
# str: __FILENAMEBASE__ will be substituted with the log dir name, any other 
# string will be used as is (root_logs will be prepended)
log_dir: __FILENAMEBASE__               
# cifar10 | stl10_48
dataset: cifar10                      
# Boolean: Whether to use class conditioning in GAN
conditioning: False                   

# int: Training batch size
batch_size: 64                        
# int: Validation batch size
batch_size_inference: 64              
# int: Total number of training steps
num_training_steps: 100000            
# int: Interval for loss logging
num_log_loss_steps: 10                
# int: Interval for images saving
num_log_images_steps: 5000            
# int: Interval for singular values analysis
num_log_sv_steps: 5000                
# int: Interval for checkpointing
num_checkpoint_steps: 5000           
# int: Interval for validation
num_validation_images: 50000         

# sngan: Base CNN template
model_name: sngan                     
# int: 32 for cifar10, 48 for stl10_48
model_preset: 32                      
# linear: Linear decay from the base value down to 0 after num_training_steps
lr_sched: linear                      
# hinge: GAN hinge loss used
loss_name: hinge                      
# Boolean: Whether to use the canonical ("reduced" in the paper) Householder 
# parameterization
use_canonical_householder: True       

# str: GAN metric name used for monitoring model performance
fidelity_leading_metric_name: inception_score_mean  
# Dict: extra kwargs for GAN metrics. See API documentation:
# https://torch-fidelity.readthedocs.io/en/latest/usage_api.html
# https://torch-fidelity.readthedocs.io/en/latest/api.html
fidelity_kwargs:                      
    batch_size: 32         
    isc: True              
    fid: True              
    kid: True              
    ppl: False             
    model_z_size: 128      
    verbose: False         
# normal: GAN noise type
z_type: normal                        
# lerp | slerp_any: GAN latent code interpolation type
z_interp_mode: slerp_any              
# int: How many rows to keep in singular values visualization
vis_truncate_singular_values: 32      

# Dict: Extra kwargs for the Generator model
g_kwargs: {}                          
# Dict: Extra kwargs for the Discriminator model
d_kwargs: {}                          

# Boolean: Whether to use exponential moving average on the Generator weights
g_ema: False                          

# regular | spectral_norm_pytorch: Whether to use standard CNN convs within 
# the Generator (which can later be reparameterized), or spectrally-
# normalized (Miyato 2018)
g_ops_name: regular                   
# Boolean: Whether to perform reparameterization of the Generator (needs to be 
# 'regular')
g_ops_use_factory: True               
# svdp | sttp: Type of Generator reparameterization
g_ops_factory_name: svdp               
# householder: Stiefel parameterization type of the Generator
g_ops_factory_stiefel_name: householder  
# int: Maximum rank of any tensor parameterized by the Generator factory 
# (larger is more expressive and slower)
g_ops_factory_rank: 64                 
# Dict: Extra kwargs for the Generator factory
g_ops_factory_kwargs:                 
    # Boolean: If true, all parameterized matrices' singular values are 1
    spectrum_eye: False               
    # int: RNG reproducibility seed
    init_seed: 2020                   
    # qr_eye_randn | qr_randn: Whether to use noisy identity or random 
    # initialization or orthonormal matrices
    init_mode: qr_eye_randn           
    # float: std of random noise used during initialization
    init_std: 1e-4                    
# List: Names of torch.nn.Module names (relative to the model root) not subject 
# to reparameterization in the Generator
g_ops_factory_ignored:               
    - noise_to_conv

# int: Interval for spectral compensation for unparameterized Generator 
# (0 - never)
g_spectral_compensation_frequency: 0  
# d_optimal | divergence: Type of Generator singular values regularizer when 
# spectrum_eye is False
g_spectral_penalty: d_optimal         
# float: Generator singular values regularizer loss weight
g_spectral_penalty_weight: 1.0        

# adam | sgd: Type of optimizer used for the Generator
g_optimizer: adam                     
# Dict: Extra kwargs for the Generator optimizer (must at least have lr)
g_optimizer_kwargs:                   
    lr: 2e-4          
    betas: [0.0, 0.9] 

# regular | spectral_norm_pytorch: Whether to use standard CNN convs within the 
# Discriminator (which can later be reparameterized), or spectrally-normalized 
# (Miyato 2018)
d_ops_name: regular
# Boolean: Whether to perform reparameterization of the Discriminator (needs to 
# be 'regular')
d_ops_use_factory: True
# svdp | sttp: Type of Discriminator reparameterization
d_ops_factory_name: svdp
# householder: Stiefel parameterization type of the Discriminator
d_ops_factory_stiefel_name: householder
# int: Maximum rank of any tensor parameterized by the Discriminator factory 
# (larger is more expressive and slower)
d_ops_factory_rank: 64
# Dict: Extra kwargs for the Discriminator factory
d_ops_factory_kwargs:
    # Boolean: If true, all parameterized matrices singular values are 1
    spectrum_eye: False
    # int: RNG reproducibility seed
    init_seed: 2020
    # qr_eye_randn | qr_randn: Whether to use noisy identity or random 
    # initialization or orthonormal matrices
    init_mode: qr_eye_randn
    # float: std of random noise used during initialization
    init_std: 1e-4
# List: Names of torch.nn.Module names (relative to the model root) not subject 
# to reparameterization in the Discriminator
d_ops_factory_ignored: []

# int: Interval for spectral compensation for unparameterized Discriminator 
# (0 - never)
d_spectral_compensation_frequency: 0
# d_optimal | divergence: Type of Discriminator singular values regularizer 
# when spectrum_eye is False
d_spectral_penalty: d_optimal
# float: Discriminator singular values regularizer loss weight
d_spectral_penalty_weight: 1.0

# adam | sgd: Type of optimizer used for the Discriminator
d_optimizer: adam
# Dict: Extra kwargs for the Discriminator optimizer (must at least have lr)
d_optimizer_kwargs:
    lr: 2e-4
    betas: [0.0, 0.9]
    
# int: Number of Discriminator updates per one Generator update
d_step_repeats: 5                     
```

### GAN Configuration Discussion

The GAN setting consists of two networks: the Generator and the Discriminator, each of which can be reparameterized or
otherwise regularized independently of another. The following examples are showing different possibilities of 
parameterizations, each of which can be found in our preconfigured experiments:

#### Unconstrained Discriminator 

```yaml
d_ops_name: regular
d_ops_use_factory: False
d_spectral_compensation_frequency: 0
d_spectral_penalty: Null
```

#### Spectral Normalization of the Discriminator (SNGAN)

```yaml
d_ops_name: spectral_norm_pytorch
d_ops_use_factory: False
d_spectral_compensation_frequency: 0
d_spectral_penalty: Null
```

#### Spectral Compensation of the Discriminator (SRGAN), applied every step

```yaml
d_ops_name: regular
d_ops_use_factory: False
d_spectral_compensation_frequency: 1
d_spectral_compensation_kwargs:
    normalize: True
d_spectral_penalty: Null
```

#### SVDP-C Discriminator with rank 64

```yaml
d_ops_name: regular
d_ops_use_factory: True
d_ops_factory_name: svdp
d_ops_factory_stiefel_name: householder
d_ops_factory_rank: 64
d_ops_factory_kwargs:
    spectrum_eye: True
    init_seed: 2020
    init_mode: qr_eye_randn
    init_std: 1e-4
d_ops_factory_ignored: []
d_spectral_compensation_frequency: 0
d_spectral_penalty: Null
```

#### SVDP-R Discriminator with rank 64 and regularization weight 1.0

```yaml
d_ops_name: regular
d_ops_use_factory: True
d_ops_factory_name: svdp
d_ops_factory_stiefel_name: householder
d_ops_factory_rank: 64
d_ops_factory_kwargs:
    spectrum_eye: False
    init_seed: 2020
    init_mode: qr_eye_randn
    init_std: 1e-4
d_ops_factory_ignored: []
d_spectral_compensation_frequency: 0
d_spectral_penalty: d_optimal
d_spectral_penalty_weight: 1.0
```

#### STTP-L Discriminator with rank 64

```yaml
d_ops_name: regular
d_ops_use_factory: True
d_ops_factory_name: sttp
d_ops_factory_stiefel_name: householder
d_ops_factory_rank: 64
d_ops_factory_kwargs:
    spectrum_eye: False
    init_seed: 2020
    init_mode: qr_eye_randn
    init_std: 1e-4
d_ops_factory_ignored: []
d_spectral_compensation_frequency: 0
d_spectral_penalty: Null
```

## Classification Configuration Template

The following config template should be used with `train_imgcls.py` training script:

```yaml
# imgcls: Experiment type
experiment: imgcls
# str: __FILENAMEBASE__ will be substituted with the file name, any other 
# string will be used as is
experiment_name: __FILENAMEBASE__       
# str: __FILENAMEBASE__ will be substituted with the log dir name, any other 
# string will be used as is (root_logs will be prepended)
log_dir: __FILENAMEBASE__
# cifar10
dataset: cifar10

# int: Training batch size
batch_size: 128
# int: Total number of training steps
num_training_steps: 100000
# int: Interval for loss logging
num_log_loss_steps: 10
# int: Interval for singular values analysis
num_log_sv_steps: 5000
# int: Interval for checkpointing
num_checkpoint_steps: 5000
# linear: Linear decay from the base value down to 0 after num_training_steps
lr_sched: linear

# int: How many rows to keep in singular values visualization
vis_truncate_singular_values: 32
# Boolean: Whether to use the canonical ("reduced" in the paper) Householder 
# parameterization
use_canonical_householder: True

# wresnet_cifar: Base CNN template
model_name: wresnet_cifar
# Unused
model_name_specific: Null
# Dict: Extra kwargs for the base model
model_name_specific_kwargs: {}

# Boolean: Whether to perform reparameterization of the model
ops_use_factory: True
# svdp | sttp: Type of model reparameterization
ops_factory_name: sttp
# householder: Stiefel parameterization type of the model
ops_factory_stiefel_name: householder
# int: Maximum rank of any tensor parameterized by the model factory 
# (larger is more expressive and slower)
ops_factory_rank: 256
# Dict: Extra kwargs for the model factory
ops_factory_kwargs:
    # Boolean: If true, all parameterized matrices singular values are 1
    spectrum_eye: False
    # int: RNG reproducibility seed
    init_seed: 2020
    # qr_eye_randn | qr_randn: Whether to use noisy identity or random 
    # initialization or orthonormal matrices
    init_mode: qr_eye_randn
    # float: std of random noise used during initialization
    init_std: 1e-4
# List: Names of torch.nn.Module names (relative to the model root) not subject 
# to reparameterization in the model
ops_factory_ignored: []

# d_optimal | divergence: Type of model's singular values regularizer when 
# spectrum_eye is False
spectral_penalty: d_optimal
# float: Model singular values regularizer loss weight
spectral_penalty_weight: 1.0

# adam | sgd: Type of optimizer
optimizer: sgd
# Dict: Extra kwargs the optimizer (must at least have lr)
optimizer_kwargs:
    lr: 0.1
    momentum: 0.9
    dampening: 0
    weight_decay: 0.0001
    nesterov: False
```

### Classification Configuration Discussion

The classification setting consists of just the classification network. The examples discussed above in relation to 
various GAN Discriminator reparameterizations apply identically to the classification setting. 
