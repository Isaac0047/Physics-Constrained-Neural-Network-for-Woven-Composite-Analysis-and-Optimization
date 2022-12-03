##This folder contains the implementation of baseline models we considered in this research (Only for inverse prediction (BDPs))

Trained CNN_models (FDP, paper Figure 7): Weave_CNN_model_E7.h5
                                          Weave_bimat_CNN_mat_vec_update.h5

Genetic Algorithm (paper Figure C.8): Weave_single_mat_ga_model.py (single material, need to use single material CNN (Weave_CNN_model_E7.h5))
                                      Weave_bimat_ga_model.py (bi material, need to use single material CNN (Weave_bimat_CNN_mat_vec_update.h5))
                   
Encoder-Decoder (paper Figure C.4 & C.5): Weave_bimat_Deconv_Pattern.py (bi material, pattern prediction)
                                          Weave_bimat_Deconv_mat_vec.py (bi material, material sequence prediction)
                 
GAN (paper Figure C.6 & C.7): Weave_bimat_GAN_Pattern.py (bi material, pattern prediction)
                              Weave_bimat_GAN_mat_vec.py (bi material, material sequence prediction)
