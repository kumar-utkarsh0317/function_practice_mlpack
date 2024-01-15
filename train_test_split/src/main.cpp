#include <iostream>
#include<armadillo>
// #include "function_decleration.hpp"
using namespace std;
#include"utklearn.hpp"
using namespace utk;

int main(){
    arma::mat data, X_train, X_test, y_train, y_test;
    arma::field<std::string> headers;
    bool load_status = data.load(arma::csv_name("/home/kumar/Desktop/mlpack_function_practice/train_test_split/data/mainData.csv", headers, arma::csv_opts::trans));

    // data = arma::trans(data);
    train_test_split(data, X_train, X_test, y_train, y_test, 20);

    // cout<<"rows and cols of data ("<<data.n_rows<<" ,"<<data.n_cols<<")\n\n";
    // cout<<"rows and cols of X_train ("<<X_train.n_rows<<" ,"<<X_train.n_cols<<")\n\n";
    // cout<<"rows and cols of y_train ("<<y_train.n_rows<<" ,"<<y_train.n_cols<<")\n\n";
    // cout<<"rows and cols of X_test ("<<X_test.n_rows<<" ,"<<X_test.n_cols<<")\n\n";
    // cout<<"rows and cols of y_test ("<<y_test.n_rows<<" ,"<<y_test.n_cols<<")\n\n";

    return 0;
}