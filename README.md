# Adapative Rate Task-Oriented Vector Quantization (ARTOVeQ)

ARTOVeQ is designed to address the challenges of learned compression mechanisms that struggle to adapt their resolution over time-varying links. Most existing DNN-aided compression algorithms operate in a single-rate manner. In the context of remote inference, this introduces two notable challenges when communicating over time-varying links:
(i) Once trained, the model's compression rate can not be altered
(ii) Inference can only begin after all the compressed features arrive at the inferring device

Our work tackles these challenges (i-ii) by designing a learned compression mechanism that is independent of the network architecture and focuses on the quantization process itself.
