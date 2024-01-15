# Setting the variables
CXX = g++
HEADER_PATH = -I/home/kumarutkarsh/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include
LIB_DEPENDENCIES = -larmadillo
CSV_DATA_PATH = "/home/kumarutkarsh/Desktop/mlpack_function_practice/datasets/bodyfat.csv"

# The all target is used to specify the default target that should be built when you run make without specifying a target.
all: 1_nn_train 

# Target for building the executable '1_nn_train'
# her we define how targets should be made
1_nn_train: 1_nn_train.cpp
	$(CXX) $(HEADER_PATH) 1_nn_train.cpp -o 1_nn_train -lutklearn $(LIB_DEPENDENCIES) 



# Target for running the executable
# If you run make run, it will check whether the 1_nn_train executable is up-to-date. If the 1_nn_train executable is not up-to-date (either it doesn't exist or the source file 1_nn_train.cpp has been modified), it will first build the 1_nn_train target and then execute the 
# make run will first make the executable and then run it
run: 1_nn_train 
	./1_nn_train $(CSV_DATA_PATH)

# Target for cleaning up generated files
clean:
	rm -f 1_nn_train
	rm -f 3_lr_rough
