# eigendigits
Project description:
Using PCA to find the eigen digits (or principal components) of given handwritten digits (training set) and classify other given handwritten digits (test set) and measure the accuracy. Also, vary different paramters like number of eigen vectors, training set size etc and analyze its effect on the accuracy.  

This folder contains:
1. src		- the source code for the experiments (in Python)
2. data 	- containing the input data (training and test sets)
3. report	- contains the final report as well as different plots and images used in the report

To run the program:
1. single run: Triggers the experiment to classify the test set and prints the accuracy for the given parameters. 
Usage:	python -u main.py single <training_set_size> <test_set_size> <knn_size> <num_eigen_vectors> <difficulty> <experiment_number> <knn_distance_type>
		knn_size 		- number of nearest neighbours to use in classification
		difficulty 		- 0 (Hard), 1 (Easy)
		experiment_number 	- some integer number, directories are automatically created to store output data
		knn_distance_type 	- One of 'Euclidean', 'Cosine' or 'Manhattan'

2. batch run: Triggers the experiment for a hard-coded set of parameters and plots all the graphs in the corresponding directory created (with suffix of experiment_number)
Usage:	python -u main.py batch [<experiment_number>] [<number_of_trials_for_each_sub_experiment>] [<'test'>]
		All paramters are optional
		If 'test' is specified, very smaller values are chosen for paramters (just to see if end-to-end program works, no analysis can be done from this)

Apart from this, the code is sufficiently commented and method names and variable names are self-explanatory. 
