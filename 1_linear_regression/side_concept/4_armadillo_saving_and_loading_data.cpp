#include <iostream>
#include <armadillo>

using namespace arma;

int main() {
    arma::mat data, train, test;
    arma::field<std::string> headers;
    bool load_status = data.load(arma::csv_name("../Folds5x2_pp.csv", headers, arma::csv_opts::trans));
    const int data_cols = data.n_cols;
    const int data_rows = data.n_rows;
    cout<<"\n\ndata rows and cols are ("<<data.n_rows<<", "<<data.n_cols<<")\n\n";

    arma::rowvec v = randi<arma::rowvec>(data_cols, arma::distr_param(1, 2));
    cout<<"\n\nnumber of element in the v is "<<v.n_elem<<"\n\n"<<endl;

    //iterating through the vector element
    for (int i = 0; i < data_cols; i++){
        if(v(i) == 1){
            train = arma::join_horiz(train, data.col(i));
        }else{
            test = arma::join_horiz(test, data.col(i));
        }
    }
    cout<<"\n\ntrain rows and cols are ("<<train.n_rows<<", "<<train.n_cols<<")\n\n";
    cout<<"\n\ntest rows and cols are ("<<test.n_rows<<", "<<test.n_cols<<")\n\n";

    //loading the data to a different file
    bool train_saved = train.save(arma::csv_name("train.csv", headers, arma::csv_opts::trans));
    bool test_saved = test.save(arma::csv_name("test.csv", headers, arma::csv_opts::trans));
    if(train_saved){
        cout<<"train data is saved"<<endl;
    }
    if(test_saved){
        cout<<"test data is saved"<<endl;
    }


    return 0;
}
