#include <iostream>
#include<string>
#include "/home/kumar/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include/mlpack.hpp"
#include <armadillo>
#include "utklearn.hpp"
using namespace std;


int main(){
    std::string file_name = "file.txt";
    arma::field<std::string> field= {"hello", "world"};
    arma::rowvec v = {1, 2};
    arma::rowvec v1 = utk::csv_string2integer(file_name, field, v);

    v1.print("this is the row vector that we are getting");

    return 0;
}