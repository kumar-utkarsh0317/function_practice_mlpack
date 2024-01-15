#include <iostream>
#include<string>
#include "/home/kumar/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include/mlpack.hpp"
#include <armadillo>
#include "utklearn.hpp"
using namespace std;

int main(int argc, char**argv){
    if(argc != 2){
        std::cerr<<"usage: "<<argv[0]<<"<csv_training>"<<endl;
        return -1;
    }

    //loading the data 
    std::cout<<"\n/nwe are loading the from the file::"<<argv[1]<<"\n\n";
    arma::mat data, X_train, X_test, y_train, y_test;
    arma::field<std::string> headers;
    bool load_status = data.load(arma::csv_name(argv[1], headers, arma::csv_opts::trans));

    //if unable to load the file then terminate the program
    if(!load_status){
        cout<<"file "<<argv[1]<<"could not be loaded\n";
        return 1;
    }
    std::cout<<"file loaded. and the headers are::"<<"\n\n";
    
    //printing the headers
    for(string s : headers)
    {
        cout<<s<<"  ";
    }
    cout<<"\n\n";

    //removing the indexing row and 
    // giving some integer value to the classes so tha while loading the stirngs do not become zero
    data.shed_row(0);

    
    utk::train_test_split(data, X_train, X_test, y_train, y_test, 70);
    cout<<endl;

    //saving the test data
    // arma::mat test_data = arma::trans(arma::join_vert(X_test, y_test));
    // test_data.save(arma::csv_name("test_data.csv"));
    // cout<<"Test data saved in test_data.csv"<<endl;

    //////training the model
    utk::normalize_features(X_train);
    std::cout<<"we are now training our model"<<"\n\n";
    mlpack::KNN knn(X_train);

    // The matrices we will store output in.
    arma::Mat<size_t> resultingNeighbors;
    arma::mat resultingDistances;

    knn.Search(3, resultingNeighbors, resultingDistances);
    cout<<"rows and col of resulting neighbout "<<resultingNeighbors.n_rows<<" "<<resultingNeighbors.n_cols<<endl;
    // resultingNeighbors.print("resulting neighbour\n");
    // resultingDistances.print("resulting distances");


    /////saving the model to model.xml
    // std::cout<<"now we are saving the model...\n";
    // if (!mlpack::data::Save("1_lr_train.xml", "trained_model", lr)){
    //     std::cerr<< "could not save the model to model.xml file \n";
    //     return 1;
    // }
    // std::cout<<"the model data is saved to model.xml \n";

    resultingNeighbors.save(arma::csv_name("resultingNeighbors.csv"));
    cout<<"********************************train_complete************************\n\n";
    return 0;
}

