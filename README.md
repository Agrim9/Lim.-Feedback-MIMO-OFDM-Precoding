# Limited Feedback MIMO Precoding

Codes for the predictive quantization and joint time frequency interpolation strategy for interpolation of MIMO precoding matrices. 

* [Conference pre-print](./Manuscripts/conf_vs.pdf)
* [Extended journal version pre-print](./Manuscripts/journal_vs.pdf)

Running these codes require the package [py-itpp](https://github.com/vidits-kth/py-itpp) installed with `python-2.7`, alongside standard packages like `pdb, numpy, scipy, matplotlib`


## Code Description

* **Codebooks Generation**: Generate independent codebooks for quantization of matrices residing on the Stiefel Manifold using the code [indep\_stiefel\_quant.py](./indep_stiefel_quant.py). 
> Used for initialization of the hopping and time based predictive quantization. Also used to compare the interpolation approaches (Cayley and Geodesic). 

 Also, generate the codebook for singular values. This is done using the code [eigenvalue\_quant.py](./eigenvalue_quant.py). 

* **Interpolation Comparison**: Now, once the independent codebooks have been generated, you can move on towards simulating the comparison between the Cayley Exponential map interpolation versus the Unitary Geodesic interpolation. This is done via the code [interpolation\_comparison.py](./interpolation\_comparison.py). You can generate the results similar to Fig. 6,7 using this code with the given codebooks (in the "[Independent Quantization Codebooks folder](./Codebooks/Independent_qt)"). To generate the ideal unquantized BER curves, set `use_cb` to `False`. Also, to obtain the maximum capacity, use the code [max\_cap.py](./max_cap.py) and set SNR to 0 dB (since the achievable rate result has been generated for 0 dB SNR in the `interpolation_comparison` code).

* **Tangent Space Codebook Generation**: Moving on towards the joint time-frequency interpolation and predictive quantization scheme, the first step is to generate the codebooks for the tangent space quantization scheme initialization.
 We consider 10,000 independent channel  evolutions. For each one, we use the evolution of the channel over 10 OFDM frames’ duration to obtain a reliable prediction from both methods. We collect the vectors representing the skew Hermitian tangents for vector 
 quantisation (recall that Cayley Exp. map has skew Hermitian matrices as images 
 (Section III of paper), which form a vector space). This is done to get 10,000 independent predictions that give the same number of independent vectors in the collection. This collection is generated via the [vec\_collec_gen.py](./vec_collec_gen.py) code. After this, use the [codebook\_gen.py](./codebook_gen.py) code to apply k-means algorithm to get a 6 bit base codebook for, for both the hopping based and time based schemes. Sample codebooks generated here are in the [Predictive Quantization Codebooks](./Codebooks/Pred_qt) folder.

* **Joint time-frequency interpolation and predictive quantization**: Now once we have the base codebook for tangent spaces and the independent codebooks for initialisation purposes, we can proceed ahead and generate the precoders corresponding to time based and hopping based schemes. Using the code [hop\_pred.py](./hop_pred.py), one generates precoders corresponding to 100 channel evolutions for 10 independent channel realizations (i.e. fill 10 independent 100*`num_subcarriers` time-frequency bins matrix with generated precoders), using both the time based and hopping based schemes. These precoders are stored as `.npy` files in [Precoders\_generated](./Precoders_generated) folder. The folder currently contains sample precoders generated for ITU Pedestrian-A Channel. Using [eval.py](./eval.py), you can get the qtisation error with time, achievable rate and BER results, viz. the figures 8,9 in the paper.

### Helping Codes

* **Plotting code**: The python script [plot\_res.py](./plot_res.py) contains the codes used to plot the results and save svg files corresponding to them. The file also contains the results obtained by us in raw data form as well. 

* **BER and Achievable Rate codes**: The python script [sumrate\_BER.py](./sumrate_BER.py) contains the codes used to obtain the BER and achievable rate

* **MIMO TDL Channel**: The python script [mimo\_tdl\_channel.py](./mimo_tdl_channel.py) contains codes for channel generation, lifting and precoder prediction

---

Feel free to mail at `agrim@ee.iitb.ac.in` for any further clarifications regarding the project. 
