#include <iostream>
#include <vector>
#include<string>
#include <mlpack.hpp>
#include <ensmallen.hpp>
#include <armadillo>
#include "utklearn.hpp"
using namespace std;

int main(int argc, char**argv){
    if(argc != 2){
        std::cerr<<"usage: "<<argv[0]<<"<csv_training>"<<endl;
        return -1;
    }

    //loading the data 
    std::cout<<"we are loading the from the file::"<<argv[1]<<std::endl;
    arma::mat predictor;
    arma::mat data, X_train, X_test, y_train, y_test;
    arma::field<std::string> headers;


    // bool load_status = data.load(arma::csv_name(argv[1], headers, arma::csv_opts::trans));
    mlpack::data::Load(argv[1], data);
    // data = data.t();      already transposed we dont need to do it

    //putting the fat data in the last row and saving them
    arma::rowvec body_fat = data.row(1);
    data.shed_row(1);
    data = arma::join_vert(data, body_fat);

    //no need to transpose
    data = data.t();
    // mlpack::data::Save("bodyfat.csv", data);

    utk::train_test_split(data, X_train, X_test, y_train, y_test, 70);

    //if unable to load the file then terminate the program
    // if(!load_status)
    // {
    //     cout<<"file "<<argv[1]<<"could not be loaded\n";
    //     return 1;
    // }
    std::cout<<"file loaded. and the headers are::"<<std::endl;
    // for(string header : headers)
    // {
    //     cout<<header<<" "<<endl;
    // }
    // cout<<"\n\n";

    // //saving the test data
    // arma::mat test_data = arma::trans(arma::join_vert(X_test, y_test));
    // test_data.save(arma::csv_name("test_data.csv"));
    // cout<<"Test data saved in test_data.csv"<<endl;

    // // scaling the data
    mlpack::data::MinMaxScaler scaleX, scaleY;
    scaleX.Fit(X_train);
    scaleX.Transform(X_train, X_train);
    scaleX.Transform(X_test, X_test);

    scaleY.Fit(y_train);
    scaleY.Transform(y_train, y_train);
    scaleY.Transform(y_test, y_test);

    // //creting the model
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::HeInitialization> model;
    model.Add<mlpack::Linear>(64);
    model.Add<mlpack::LeakyReLU>();
    model.Add<mlpack::Linear>(128);
    model.Add<mlpack::LeakyReLU>();
    model.Add<mlpack::Linear>(64);
    model.Add<mlpack::LeakyReLU>();
    model.Add<mlpack::Linear>(1);

    


    // // training the model
    model.Train(
        X_train,
        y_train,
        optimizer,
        ens::PrintLoss(),
        ens::ProgressBar(40),
        ens::EarlyStopAtMinLoss(20)
    );

    // //training the model
    // std::cout<<"we are now training our model"<<std::endl;
    // mlpack::LinearRegression lr(X_train, y_train);
    // std::cout << "Computed error: "<< lr.ComputeError(X_train, y_train) << '\n';

    // //saving the model to model.xml
    // std::cout<<"now we are saving the model...\n";
    // if (!mlpack::data::Save("1_lr_train.xml", "trained_model", lr)){
    //     std::cerr<< "could not save the model to model.xml file \n";
    //     return 1;
    // }
    // std::cout<<"the model data is saved to model.xml \n";
    // cout<<"********************************train_complete************************\n\n";

    cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;

    std::vector<size_t> v = model.InputDimensions();
    for(auto element : v)
    {
        cout<<element<<" ";
    }
}

ens::Adam optimizer(
    0.05,    // Step size of the optimizer.
    32,      // Batch size. Number of data points that are used in each iteration.
    0.9,     // Exponential decay rate for the first moment estimates.
    0.999,   // Exponential decay rate for the weighted infinity norm estimates.
    1e-8,    // Value used to initialise the mean squared gradient parameter.
    X_train.n_cols * 30,  // Max number of iterations.
    1e-8,    // Tolerance.
    true);   // Shuffle the data before each epoch.