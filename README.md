# Physics-Constrained-Neural-Network-for-the-Analysis-and-Feature-based-Optimization-of-Woven-Composite

12/03/2022 (Latest Update)

By Haotian Feng, Sabarinathan P Subramaniyan and Pavana Prabhakar

Department of Mechanical Engineering
University of Wisconsin - Madison

This folder provides the code for implementing our proposed Neural Network frameworks as well as the training dataset. The details of the paper can be found at https://arxiv.org/pdf/2209.09154.pdf

## Implementation
In this repository, all frameworks are implemented using Tensorflow and Keras. To run the PCNN or baseline model frameworks, please download the code in the 'PCNN_Implementation' or 'Baseline_Model_implementation' folder, and also download the dataset from 'Dataset' folder into the same repository. The dataset is stored in .npy format, trained Neural Network is saved in .h5 format, and the main framework are implemented in .py.

>The DCNN framework for FDP is shown in paper Figure 7. 
>
>PCNN framework (Material to Pattern) is shown in paper Figure 8. 
>
>PCNN framework (Pattern to Material) is shown in paper Figure 9. 

## Citation
Please cite our paper as:

@misc{https://doi.org/10.48550/arxiv.2209.09154,
  doi = {10.48550/ARXIV.2209.09154},
  
  url = {https://arxiv.org/abs/2209.09154},
  
  author = {Feng, Haotian and Subramaniyan, Sabarinathan P and Prabhakar, Pavana},
  
  keywords = {Applied Physics (physics.app-ph), Machine Learning (cs.LG), FOS: Physical sciences, FOS: Physical sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Physics-Constrained Neural Network for the Analysis and Feature-Based Optimization of Woven Composites},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

