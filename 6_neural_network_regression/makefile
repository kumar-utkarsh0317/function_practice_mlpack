# Setting the variables
CXX = g++
HEADER_PATH = -I/home/kumarutkarsh/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include
LIB_DEPENDENCIES = -larmadillo
CSV_DATA_PATH = "/home/kumarutkarsh/Desktop/mlpack_function_practice/datasets/bodyfat.csv"

# The all target is used to specify the default target that should be built when you run make without specifying a target.
all: train 

# Target for building the executable 'train'
# her we define how targets should be made
train: train.cpp
	$(CXX) $(HEADER_PATH) train.cpp -o train  -larmadillo



# Target for running the executable
# If you run make run, it will check whether the train executable is up-to-date. If the train executable is not up-to-date (either it doesn't exist or the source file train.cpp has been modified), it will first build the train target and then execute the 
# make run will first make the executable and then run it
run: train 
	./train

# Target for cleaning up generated files
clean:
	rm -f train
