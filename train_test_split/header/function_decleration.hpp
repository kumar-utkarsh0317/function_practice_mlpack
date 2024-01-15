#ifndef function_decleration_HPP
#define function_decleration_HPP
#include <armadillo>
#include<istream>
using namespace std;


void train_test_split(arma::mat& data, arma::mat& X_train, arma::mat& X_test, arma::mat& y_train, arma::mat& y_test, float percent);

#endif