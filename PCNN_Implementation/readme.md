##This folder contains the implementation of our proposed PCNN (To run these models, please download the dataset from dataset folder into the same repository of codes)

'weave_CNN_model_E7.h5'             -- trained Neural Net for FDP problem for single material woven composite (from Weave_CNN_model_E7.h5)

'weave_bimat_CNN_mat_vec_update.h5' -- trained Neural Net for FDP problem for bi-material woven composite (from Weave_bimat_CNN_mat_vec_update.h5)

Note: Function to call these trained models are included in the BDP problem implementations

'Weave_model_PCNN_single.py' implements the PCNN for single material woven composite (paper Figure 7)

'weave_cnn_bimat.py' implements the PCNN for bi-material woven composite (paper Figure 7)

'Weave_bimat_phy_gan_mat_to_pat_cnn.py' implements the PCNN for BDPa problem of bi-material woven composite (paper Figure 8)

'Weave_bimat_phy_gan_pat_to_mat_vec_cnn.py' implements the PCNN for BDPb problem of bi-material woven composite (paper Figure 9)
