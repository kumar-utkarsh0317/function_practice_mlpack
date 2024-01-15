#include <iostream>
#include<string>
#include <mlpack.hpp>
#include <armadillo>
#include "utklearn.hpp"
using namespace std;

//the agument in the main will help in taking input when we are running the executable file
//arg c=>number of input given at the runtime of executable 
//argv is a pointer list that contains the location of the input string
int main(int argc, char**argv){
    //here we want two argument from the command line
    //first ./executable and second file.csv
    //>>>> ./executable file.csv
    if(argc != 2){
        std::cerr<<"usage: "<<argv[0]<<"<csv_training>"<<endl;
        return -1;
        
    }

    //loading the data 
    std::cout<<"we are loading the from the file::"<<argv[1]<<std::endl;
    arma::mat predictor;
    arma::mat data, X_train, X_test, y_train, y_test;

    //declearing the field without any argument so any number of element it can store
    arma::field<std::string> headers;
    //data from the first col of the csv file will be stored in the header and rest of the data
    //will be used in the predictors
    //mlpack depends on armadillo and armadillo depends on column major policy so we 
    //are transposing the csv data
    bool load_status = data.load(argv[1], arma::csv_ascii);
    utk::train_test_split(data, X_train, X_test, y_train, y_test, 30);

    //if unable to load the file then terminate the program
    if(!load_status){
        cout<<"file "<<argv[1]<<"could not be loaded\n";
        return 1;
    }
    std::cout<<"file loaded. and the headers are::"<<headers<<std::endl;

    //saving the test data
    arma::mat test_data = arma::trans(arma::join_vert(X_test, y_test));
    test_data.save("test_data.csv");
    cout<<"Test data saved in test_data.csv"<<endl;

    //training the model
    std::cout<<"we are now training our model"<<std::endl;
    mlpack::LinearRegression lr(X_train, y_train);
    std::cout << "Computed error: "<< lr.ComputeError(X_train, y_train) << '\n';

    //saving the model to model.xml
    std::cout<<"now we are saving the model...\n";
    if (!mlpack::data::Save("1_lr_train.xml", "trained_model", lr)){
        std::cerr<< "could not save the model to model.xml file \n";
        return 1;
    }
    std::cout<<"the model data is saved to model.xml \n";
    cout<<"********************************train_complete************************\n\n";
    return 0;
}

