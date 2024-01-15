CXX = g++
HEADER_PATH = -I/home/kumarutkarsh/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include
LIB_DEPENDENCIES = -larmadillo
CSV_DATA_PATH = "/home/kumarutkarsh/Desktop/mlpack_function_practice/datasets/bodyfat.csv"

# The all target is used to specify the default target that should be built when you run make without specifying a target.
all: main 

# Target for building the executable 'main'
# her we define how targets should be made
main: main.cpp
	$(CXX) $(HEADER_PATH) main.cpp -o main $(LIB_DEPENDENCIES) 



# Target for running the executable
# If you run make run, it will check whether the main executable is up-to-date. If the main executable is not up-to-date (either it doesn't exist or the source file main.cpp has been modified), it will first build the main target and then execute the 
# make run will first make the executable and then run it
run: main 
	./main $(CSV_DATA_PATH)

# Target for cleaning up generated files
clean:
	rm -f main
