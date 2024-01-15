# Setting the variables
CXX = g++
HEADER_PATH = -I/home/kumarutkarsh/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include
LIB_DEPENDENCIES = -larmadillo
CSV_DATA_PATH = "/home/kumarutkarsh/Desktop/mlpack_function_practice/datasets/lr_.csv"

# The all target is used to specify the default target that should be built when you run make without specifying a target.
all: 1_lr_train 2_lr_test

# Target for building the executable '1_lr_train'
# her we define how targets should be made
1_lr_train: 1_lr_train.cpp
	$(CXX) $(HEADER_PATH) 1_lr_train.cpp -o 1_lr_train $(LIB_DEPENDENCIES) -lutklearn 

2_lr_test: 2_lr_test.cpp
	$(CXX) $(HEADER_PATH) 2_lr_test.cpp -o 2_lr_test -lutklearn $(LIB_DEPENDENCIES) 

3_lr_rough: 3_lr_rough.cpp
	$(CXX) $(HEADER_PATH) 3_lr_rough.cpp -o 3_lr_rough $(LIB_DEPENDENCIES)

# Target for running the executable
# If you run make run, it will check whether the 1_lr_train executable is up-to-date. If the 1_lr_train executable is not up-to-date (either it doesn't exist or the source file 1_lr_train.cpp has been modified), it will first build the 1_lr_train target and then execute the 
# make run will first make the executable and then run it
run: 1_lr_train 2_lr_test
	./1_lr_train $(CSV_DATA_PATH)
	./2_lr_test 1_lr_train.xml 

# Target for cleaning up generated files
clean:
	rm -f 1_lr_train
	rm -f 3_lr_rough
