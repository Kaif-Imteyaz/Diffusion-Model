# Diffusion Model for Image Generation

A PyTorch implementation of a denoising diffusion probabilistic model (DDPM) trained on the Stanford Cars dataset for image synthesis. This implementation is based on the foundational work by Ho et al. (2020) on DDPMs, with architectural improvements inspired by Dhariwal and Nichol (2021).

This project implements a denoising diffusion probabilistic model that learns to generate car images through a forward diffusion process and a backward denoising process. The model progressively adds Gaussian noise to training images over multiple timesteps and then learns to reverse this process to generate new samples from pure noise.

## Dataset

The implementation uses the Stanford Cars Dataset, which contains approximately 8,000 images in the training set. The dataset is organized by car classes and includes both training and test splits.

**Dataset Source:** `jutrera/stanford-car-dataset-by-classes-folder` (via Kaggle)

## Architecture

### Theoretical Foundation

The implementation follows the DDPM framework introduced by Ho et al. (2020), which establishes a connection between diffusion probabilistic models and denoising score matching with Langevin dynamics. The model is trained on a weighted variational bound that enables high-quality image generation.

### Components

1. **Forward Diffusion Process**
   - Linear beta schedule for noise variance scheduling
   - Gradual transformation of images to pure Gaussian noise over T timesteps
   - Closed-form sampling at any timestep using reparameterization
   - Based on considerations from nonequilibrium thermodynamics

2. **U-Net Backbone**
   - Encoder-decoder architecture with symmetric skip connections
   - Sinusoidal position embeddings for time conditioning
   - Time information integrated at each resolution level via learned linear projections
   - Downsampling and upsampling blocks with batch normalization
   - ReLU activation functions

3. **Training Objective**
   - Simplified objective: L1 loss between predicted and actual noise
   - Equivalent to denoising score matching over multiple noise scales
   - Model learns to predict noise ε added at each timestep t

### Network Architecture Details

**U-Net Structure:**
- Input channels: 3 (RGB)
- Time embedding dimension: 32
- Down blocks: 64 → 128 → 256 → 512 → 1024 channels
- Up blocks: 1024 → 512 → 256 → 128 → 64 channels
- Output channels: 3 (RGB)

**Block Components:**
- Convolutional layers with 3x3 kernels
- Batch normalization
- Time embedding via MLP projection
- Downsampling/upsampling via strided convolutions

## Implementation

### Requirements

```
torch
torchvision
matplotlib
numpy
kagglehub
```

### Data Preprocessing

Images undergo the following transformations:
- Resize to 64x64
- Random horizontal flip (augmentation)
- Conversion to tensor
- Normalization to [-1, 1] range

## Training Process

1. **Forward Pass:** Sample random timestep t and add corresponding noise to clean images
2. **Prediction:** U-Net predicts the noise added to the noisy image
3. **Loss Computation:** L1 loss between actual and predicted noise
4. **Optimization:** Backpropagation and parameter updates via Adam optimizer

The model saves sample images every 5 epochs to visualize training progress.

## Sampling

The trained model generates new images through iterative denoising following Algorithm 2 from Ho et al. (2020):

1. **Initialize:** Start with pure Gaussian noise x_T ~ N(0, I)
2. **Iterative denoising:** For t = T, T-1, ..., 1:
   - Predict noise ε_θ(x_t, t) using the trained U-Net
   - Compute mean μ_θ(x_t, t) using the reparameterization
   - Sample x_{t-1} from p_θ(x_{t-1} | x_t)
   - Add noise for all steps except t=1 (stochastic sampling)
3. **Output:** Return denoised image x_0

### Sampling Process

The reverse diffusion process implements ancestral sampling with learned noise prediction. Each denoising step computes:

```
x_{t-1} = μ_θ(x_t, t) + σ_t · z, where z ~ N(0, I) for t > 1
```

The posterior variance σ_t² can be fixed or learned. This implementation uses fixed variance scheduling.

## Usage

### Training

```python
# Load and prepare data
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model and optimizer
model = SimpleUnet()
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()
```

### Generation

```python
# Generate new samples
samples = sample_plot_image()
```

## Key Features

- Clean implementation of the DDPM framework from Ho et al. (2020)
- Modular U-Net architecture with skip connections
- Sinusoidal positional encodings for time conditioning
- Efficient training with GPU acceleration (CUDA support)
- Visualization of progressive noise addition in forward process
- Stochastic sampling during generation for diversity
- Progressive sample generation tracking during training


## Technical Details

### Forward Diffusion Process

Following Ho et al. (2020), the forward process gradually adds Gaussian noise according to a fixed variance schedule. At any timestep t, we can sample directly:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1 - ᾱ_t)I)
x_t = √(ᾱ_t)x_0 + √(1 - ᾱ_t)ε, where ε ~ N(0, I)
```

Where:
- `x_0` is the original clean image
- `x_t` is the noisy image at timestep t
- `ε` is noise sampled from standard Gaussian N(0, I)
- `ᾱ_t = ∏(1 - β_i)` is the cumulative product of (1 - β_i) from i=1 to t
- `β_t` is the variance schedule (linear from 0.0001 to 0.02)

### Reverse Process (Denoising)

The model learns to reverse the diffusion process by predicting the noise. The denoising step is given by:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

where μ_θ(x_t, t) = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t))
```

The model ε_θ predicts the noise, and the mean is computed using the predicted noise to denoise the image.

### Simplified Training Objective

The simplified objective from Ho et al. (2020) directly predicts noise:

```
L_simple = E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]
```

This formulation is equivalent to denoising score matching and is more stable than optimizing the full variational lower bound.

## Hardware Requirements

- GPU with CUDA support (recommended)
- Minimum 8GB VRAM for typical batch sizes
- Training time varies based on GPU capability

## References

1. **Ho, J., Jain, A., & Abbeel, P. (2020).** Denoising Diffusion Probabilistic Models. *Neural Information Processing Systems (NeurIPS)*. arXiv:2006.11239.
   - Establishing the DDPM framework
   - Introduced the simplified training objective and connection to score matching
   - [Paper](https://arxiv.org/abs/2006.11239) 

2. **Dhariwal, P., & Nichol, A. (2021).** Diffusion Models Beat GANs on Image Synthesis. *Neural Information Processing Systems (NeurIPS)*. arXiv:2105.05233.
   - Architectural improvements including attention and adaptive group normalization
   - Introduced classifier guidance for conditional generation
   - [Paper](https://arxiv.org/abs/2105.05233)
