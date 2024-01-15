#include<armadillo>
#include<iostream>
// #include "utkarsh.hpp"
#include "function_decleration.hpp"
using namespace std;

void train_test_split(arma::mat& data, 
                    arma::mat& X_train, 
                    arma::mat& X_test, 
                    arma::mat& y_train, 
                    arma::mat& y_test,
                    float percent)
                    {

    cout<<"train_test_split is on process"<<endl;
    const int n_cols = data.n_cols;
    const int n_rows = data.n_rows;

    arma::mat X, y;
    X = data.submat(0, 0, n_rows - 1, n_cols - 2);
    y = data.submat(0, n_cols - 1, n_rows - 1, n_cols - 1);
    // cout<<y.n_cols<<y.n_rows<<endl;

    arma::uvec v = arma::linspace<arma::uvec>(0, n_rows - 1, n_rows);
    v = arma::shuffle(v);

    int n_train_data = n_rows * (percent / 100); 

    //iterating through the vector element
    for (int i = 0; i < n_rows; i++){
        int row_n = v(i);
        if(i < n_train_data){
            X_train = arma::join_vert(X_train, X.row(row_n));
            y_train = arma::join_vert(y_train, y.row(row_n));
        }else{
            // cout<<"yes"<<endl;
            X_test = arma::join_vert(X_test, X.row(row_n));
            y_test = arma::join_vert(y_test, y.row(row_n));
        }
    }
    cout<<"dimension:: (rows, cols)"<<endl;
    cout<<"dimension of data:: ("<<data.n_rows<<" ,"<<data.n_cols<<")"<<endl;
    cout<<"dimension of X_train:: ("<<X_train.n_rows<<" ,"<<X_train.n_cols<<")"<<endl;
    cout<<"dimension of y_train:: ("<<y_train.n_rows<<" ,"<<y_train.n_cols<<")"<<endl;
    cout<<"dimension of X_test:: ("<<X_test.n_rows<<" ,"<<X_test.n_cols<<")"<<endl;
    cout<<"dimension of y_test:: ("<<y_test.n_rows<<" ,"<<y_test.n_cols<<")"<<endl;
    
}
