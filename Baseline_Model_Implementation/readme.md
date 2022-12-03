##This folder contains the implementation of baseline models we considered in this research (Only for inverse prediction (BDPs))

Trained CNN_models (FDP): Weave_CNN_model_E7.h5
                          Weave_bimat_CNN_mat_vec_update.h5

Genetic Algorithm: Weave_single_mat_ga_model.py (single material, need to use single material CNN (Weave_CNN_model_E7.h5))
                   Weave_bimat_ga_model.py (bi material, need to use single material CNN (Weave_bimat_CNN_mat_vec_update.h5))
                   
Encoder-Decoder: Weave_bimat_Deconv_Pattern.py (bi material, pattern prediction)
                 Weave_bimat_Deconv_mat_vec.py (bi material, material sequence prediction)
                 
GAN: Weave_bimat_GAN_Pattern.py (bi material, pattern prediction)
     Weave_bimat_GAN_mat_vec.py (bi material, material sequence prediction)
