# RAINER - GeneRic Artificial Intelligence PlaNning for CybEr-Physical PRoduction Systems

This is the complementary repository to the paper "Integrating Machine Learning into an SMT-based Planning Approach for Production Planning in Cyber-Physical Production Systems" accepted at the HYDRA Workshop at the European Conference on Artificial Intelligence (ECAI) 2023.

# Content
The repo contains an implementation of the GeneRic Artificial Intelligence PlaNning for CybEr-Physical PRoduction Systems algorithm (RAINER), and all supplementary material to reproduce the results of the paper.

Cyber-Physical Production Systems (CPPS) are highly complex systems, making the application of AI planning approaches for production planning challenging.
Most AI planning approaches require comprehensive domain descriptions, which model the functional dependencies within the CPPS.
Though, due to their high complexity, creating such domain descriptions manually is considered difficult, tedious, and error-prone.
Therefore, we propose a novel generic planning approach, which can integrate mathematical formulas or Machine Learning models into a symbolic SMT-based planning algorithm, thus shedding the need for complex manually created models. 
Our approach uses a feature-vector-based state-space representation as an interface of symbolic and sub-symbolic AI, and can identify a solution to CPPS planning problems by determining the required production steps, their sequence, and their parametrization.
We evaluate our approach on twelve planning problems from a real CPPS, demonstrating its ability to express complex dependencies within production steps as mathematical formulas or integrating ML models.

You can access the paper here (link will be provided later).

# Requirements 
Python and venv requirements cf. RAINER.yml. 

# Running RAINER
When you run the RAINER.py file, the algorithm uses the representation of the effect of the production steps as a mathematical formula by default. 
To integrate a ML model instead, the file translateSMT_FliPSi.py must be imported as 'ge'.
translateSMT_FliPSi.py generates the appropriate SMT clauses needed for integration into the satisfiability problem. 
Model training is performed using the ML_train_FliPSi.py file. 
The CPPS planning domain can be varied via input_FliPSi.py. 

# Citation
When using the code from this paper, please cite:
```
@inproceedings{Heesch2023,
  title={Integrating Machine Learning into an SMT-based Planning Approach for Production Planning in Cyber-Physical Production Systems},
  author={René Heesch and Jonas Ehrhardt and Oliver Niggemann},
  booktitle={Proceedings of the HYDRA Workshop at ECAI'23},
  year={2023}
}
```

# License
Licensed under MIT license - cf. LICENSE
