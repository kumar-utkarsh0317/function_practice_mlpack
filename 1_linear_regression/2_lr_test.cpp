#include <iostream>
#include<string>
#include "/home/kumar/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include/mlpack.hpp"
#include<utklearn.hpp>
#include <armadillo>
using namespace std;

int main(int argc, char**argv){
    if(argc != 2){
        std::cerr<<"usage: "<<argv[0]<<"<csv_training>"<<endl;
        return -1;
    }

    mlpack::LinearRegression lr;

    //second parameter is the xml element in the cereal model
    if(!mlpack::data::Load(argv[1], "trained_model", lr)){
        std::cerr<< "model loading failed\n";
        return 1;
    }

    ////loading the test data 
    arma::mat test_data;
    bool load_status = test_data.load(arma::csv_name("test_data.csv"));
    if(!load_status){
        cout<<"unable to load the test_data"<<endl;
        return -1;
    }
    test_data = arma::trans(test_data);
    arma::mat true_value = test_data.row(test_data.n_rows - 1);

    arma::mat points = test_data.submat(0, 0, test_data.n_rows - 2, test_data.n_cols -1);
    arma::rowvec predictions;

    //pasing the points
    lr.Predict(points, predictions);

    cout<<"mean absolute error while test is::"<<utk::mean_absolute_error(predictions, true_value);
    cout<<"\nmean squared error while test is::"<<utk::mean_squared_error(predictions, true_value);
    cout<<"\n*****************************test_complete**********************"<<endl;
}