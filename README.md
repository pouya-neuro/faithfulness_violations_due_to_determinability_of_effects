# faithfulness_violations_due_to_determinability_of_effects
Repository for the code and experimental results for the dissertation "Faithfulness violations due to the determinability of effects in certain classes of discrete and discretisable Markovian causal models".

The "code" directory contains a "main_script.py" file that reproduces all of the pdf files found in this repository. In terms of content:

The "visualisation" directory contains 3 pdf files corresponding to the 3 settings outlined in our dissertation. For each setting, a pdf file in this directory illustrates the performance of the skeleton search step of the PC algorithm on models without any faithfulnes violations vs. models with faithfulness violations of the types that we characterise in our paper. 

The "analyses" directory contains 6 pdf files corresponding to the 3 settings outlined in our dissertation. That is, for each setting, there are 2 pdf files in this directory. The first file compares the performance of PC's skeleton search on models without any faithfulnes violations vs. models with faithfulness violations across different model sizes. The second analysis assesses the prevalence of models with faithfuleness violations if the models are randomly sampled according to various hyper-parameters; this gives us a handle on how frequently faithfulness violations occur in our dissertation's settings (showing experimentally, that models with faithfulness violations indeed appear to "occur" with positive / i.e. non-0 probability). 

The requirements for running the scripts and reproducing the results are given in the "requirements.txt" file. 


