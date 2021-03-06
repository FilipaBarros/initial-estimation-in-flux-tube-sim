# initial-estimation-in-flux-tube-sim

This project was created under the scope of my Master Thesis: Initial Condition Estimation in Flux Tube Simulations using Machine Learning during my last year as a student's of the Integrated Master in Electrical and Computers Engineering at the Faculty of the University of Porto. 

This intends to serve as a replication package for anyone who would like to test what is presented in the thesis or that would like to apply the code to other data. Please beware that in order to use the code for other data any parth references must be chaged as well as any reference to the features and size of the data. 

- join_data.py is ment to pick one random line from each of the profiles and merge them into a single data file. 
- exploratory_data_analysis.py provides a brief analysis of the data by providing a resume statistical table of the data, a correlation heatmap, histograms for the features and pairplots for inputs vs outputs. 
- k_fold_validation.py manually tests each model you put on it for 10 folds and gives back a summary of the metrics. 
- tunner_X.py tests which configuration is best for each of the outputs and provides a comparison between these.
- fitting.py trains and fits the previously made models on tunner_X. 
- testing.py predicts new data with the preciusly fitted models.  
- dummie_models_mses.py tests 2 models: one random and one median-based in order to compare to our models' results. 
- comparison.py compares the predicted vs the real data for the outputs. 
