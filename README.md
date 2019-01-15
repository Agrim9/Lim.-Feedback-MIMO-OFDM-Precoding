# Limited-Feedback-MIMO-Precoding
Codes for the predictive quantization and joint time frequency interpolation strategy for interpolation of MIMO precoding matrices

## Code Description

* **Codebooks Generation**: Generate independent codebooks for quantization of matrices residing on the Stiefel Manifold using the code [indep\_stiefel\_quant.py](./indep_stiefel_quant.py). 
> Used for initialization of the hopping and time based predictive quantization. Also used to compare the interpolation approaches (Cayley and Geodesic). 

 Also, generate the codebook for singular values. This is done using the code [eigenvalue\_quant.py](./eigenvalue_quant.py). 

* **Interpolation Comparison**: Now, once the independent codebooks have been generated, you can move on towards simulating the comparison between the Cayley Exponential map interpolation versus the Unitary Geodesic interpolation. This is done via the code [interpolation\_comparison.py](./interpolation\_comparison.py). You can generate the results similar to Fig. 6,7 using this code with the given codebooks (in the "[Independent Codebooks folder](./Codebooks/Independent)"). To generate the ideal unquantized BER curves, set `use_cb` to `False`. Also, to obtain the maximum capacity, use the code [max\_cap.py](./max_cap.py) and set SNR to 0 dB (since the achievable rate result has been generated for 0 dB SNR in the `interpolation_comparison` code).

* **Tangent Space Codebook Generation**: Moving on towards the joint time-frequency interpolation and predictive quantization scheme, the first step is to generate the codebooks for the tangent space quantization scheme initialization.
 We consider 10,000 independent channel  evolutions. For each one, we use the evolution of the channel over 10 OFDM frames’ duration to obtain a reliable prediction from both methods. We collect the vectors representing the skew Hermitian tangents for vector 
 quantisation (recall that Cayley Exp. map has skew Hermitian matrices as images 
 (Section III of paper), which form a vector space). This is done to get 10,000 independent predictions that give the same number of independent vectors in the collection. This collection is generated via the [vec\_collec_gen.py](./vec_collec_gen.py) code. After this, use the [codebook\_gen.py](./codebook_gen.py) code to apply k-means algorithm to get a 6 bit base codebook for, for both the hopping based and time based schemes. Sample codebooks generated here are in the [Pred\_qt\_codebooks](./Codebooks/Pred_qt) folder.



## Helping Codes
