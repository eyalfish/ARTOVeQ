# Adapative Rate Task-Oriented Vector Quantization (ARTOVeQ)

ARTOVeQ is designed to address the challenges of learned compression mechanisms that struggle to adapt their resolution over time-varying links. Most existing DNN-aided compression algorithms operate in a single-rate manner. In the context of remote inference, this introduces two notable challenges when communicating over time-varying links:
- (i) Once trained, the model's compression rate can not be altered
- (ii) Inference can only begin after all the compressed features arrive at the inferring device

Our work tackles these challenges (i-ii) by designing a learned compression mechanism that is independent of the network architecture and focuses on the quantization process itself.


# Executing the code

In this repository, you will find four different Python files, each corresponding to a dedicated simulation from our paper. The codes are intended to be self-contained, meaning you can simply download the files and run them without needing external dependencies. However, it is important to note that while each file may seem similar at first glance, there are some fundamental differences worth explaining.

## Available Files:
`adaptivecb.py`

This file serves as the main simulation for this project, with all other simulations based on it.

Inside, you will find the entire Deep Learning pipeline, including data preparation and preprocessing, the adaptive quantizer, and the training function. This is followed by a script that trains the model and performs inference. The file includes three flag variables: `use_VQVAE`, `use_split`, and `mixed_resolution`. Each flag variable corresponds to a specific simulation setting as described in the research article.

These flag variables determine the forward pass through the ARTOVeQ model:
  - If all the flag variables are set to `false`, the model uses the standard `forward` function.
  - If `use_VQVAE` is set to `true`, the model uses the `forward_with_vqvae` function.


`successive_refinement.py`

Termed Progressive ARTOVeQ, the script itself is very similar to `adaptivecb.py`, but does not contain any flag variables for different simulations and the construction of the codebook does not adhere to the nested structure as in ARTOVeQ.

For progressive ARTOVeQ, the codebook vectors each are extended by a single bit following a Minkowski sum. If \mathcal{Q}_l  represents the codebook at bit-rate $l$ then the extended codebook is


  \mathcal{Q}_{l+1} = \mathcal{Q}_{l} + \left\{\tilde{\boldsymbol{e}}^{(l)}_1, \tilde{\boldsymbol{e}}^{(l)}_2\right\}


 `quantization_with_kmeans`

`ResidualVQVAE.py`

  ### How to run the code

  To run the code simply run each `.py` file. Adjust the parameters in the command line. You will find default values under the Globals&Hyperparameter and additional parameters before the training and inference


