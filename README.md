# faithfulness_violations_due_to_determinability_of_effects
Repository for code for the dissertation "Faithfulness violations due to the determinability of effects in certain classes of discrete and discretisable Markovian causal models".

The "code/" directory contains a "main_script.py" document that reproduces all of the pdf files found in this repository. In terms of content:

The "visualisations/" directory containts 3 pdf files corresponding to the 3 settings outlined in our disseration. For each setting, a pdf file in this directory illustrates the performance 
of the skeleton search step of the PC algorithm on models without any faithfulnes violations vs. models with faithfulness violations of the types we characterise. 

The "analyses/" directory contains 6 pdf files corresponding to the 3 settings outlined in our dissertation. That is, for each setting, there are two pdf files in this directory. The first file compares the performance of PC's skeleton search on models without any faithfulnes violations vs. models with faithfulness violations across different model sizes. The second analysis assesses the prevalence of models with faithfuleness violations if the models are randomly sampled according to variouss hyper-parameters; this gives us a handle on how frequently faithfulness violations occur in our dissertation's settings (showing experimentally, that models with faithfulness violations, indeed appear to "occur" with positive / i.e. non-0 probability). 

The requirements for running the scripts and reproducing the results are given in "requirements.txt" file. 


