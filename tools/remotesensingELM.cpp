#include <string>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <glog/logging.h>

#include "caffe/caffe.hpp"
using namespace std;
using namespace caffe;

void load_features(float** in,std::string filename){
	float data[800][65];
	*in = (float*)malloc(sizeof(float) * 800 * 8);
	ifstream file(filename);
	for( int row = 0 ; row < 800 ; row++ ){
		string line;
		getline(file, line);
		if( !file.good() )
			break;
		stringstream iss(line);

		for( int col = 0 ; col < 65 ; col++ ){
			string val;
			getline( iss, val, ' ');
			if( !iss.good() )
				break;
			stringstream convertor(val);
			convertor >> data[row][col];
			(*in)[row*65 + col] = data[row][col];
		}
	}

	// for( int row = 0 ; row < 800 ; row++){
	// 	for( int col = 0 ; col < 65 ; col++){
	// 		cout << data[row][col];
	// 		cout << " , ";
	//
	// 	}
	// 	cout << endl;
	// }

}

void load_labels(float** in,string filename){
	int labels[800];
	*in = (float*)malloc(sizeof(float) * 800 * 8);
	ifstream file(filename);
	for( int row = 0 ; row < 800 ; row++ ){
		string line;
		std::getline(file, line);
		if( !file.good() )
			break;
		stringstream iss(line);
		iss >> labels[row];
	}
	float pro_labels[800][8];
	for ( int row = 0 ; row < 800 ; row++ ){
		for ( int col = 0 ; col < 8 ; col ++){
			if( (int)labels[row] == (col+1) )
				pro_labels[row][col] = 1;
			else
				pro_labels[row][col] = -1;
			(*in)[row*8 + col] = pro_labels[row][col];
		}
	}

	// for( int row = 0 ; row < 800 ; row++){
	// 	for( int col = 0 ; col < 8 ; col++){
	// 		// cout << pro_labels[row][col];
	// 		// cout << " , ";
	// 	}
	// 	// cout << endl;
	// }


}

void print_output(float** out, int r, int c){
	for( int row = 0 ; row < r ; row++ ){
		for( int col = 0 ; col < c ; col++ ){
			cout << (*out)[row*c + col];
		}
		cout << endl;
	}
}

void dnn_fwd(float* in, float* out, Net<float>* net){
	cout << "going to get input blobs" << endl;
	vector<caffe::Blob<float>*> in_blobs = net->input_blobs();
	cout << "got input blobs" << endl;
	vector<caffe::Blob<float>*> out_blobs;
	// vector<caffe::Blob<float>*> ls_blobs;
	float loss;

	cout << "going to set data" << endl;
		in_blobs[0]->set_cpu_data(in);
	cout << "setted in_blob[0]" << endl;
  	in_blobs[1]->set_cpu_data(out);
  	cout << "setted in_blob[1]" << endl;
  	out_blobs = net->ForwardPrefilled(&loss);
  	// Layer<float>* layer = net->layers_[3].get();
  	// ls_blobs = layer->blobs_[0]->cpu_data();
  	cout << "ForwardPrefilled finished"<< endl;

  	// memcpy(output, out_blobs[0]->cpu_data(), sizeof(float) * 800 * 8);
  	// memcpy(weights, layer->blobs_[0]->cpu_data(), sizeof(float) * 2048 * 8);

}
int main(int argc, char** argv){
	Caffe::set_mode(Caffe::CPU);
	string network(argv[1]);
	string features(argv[2]);
	string labels(argv[3]);
	// float* input;
	float a[4*4] = {
		1.44, -7.84, -4.39,  4.53,
			1.44, -7.84, -4.39,  4.53,
		-9.96, -0.28, -3.24,  3.83,
		1.44, -7.84, -4.39,  4.53
	};
	float b[4*2] = {
		8.58, 9.35,
			8.58, 9.35,
			8.26, -4.43,
			8.58, 9.00
	};
	// float* exp_out;
	// float* elm_output = (float*)malloc(sizeof(float) * 2 * 2);
	// float* ls_weights = (float*)malloc(sizeof(float) * 2048 * 8);
	// load_features(&input, features);
	// load_labels(&exp_out, labels);
	// for( int i = 0 ; i < 800 ; i++ ){
 //    	for ( int j = 0 ; j < 65 ; j++){
 //     	   std::cout << input[i*65+j] << " , ";
 //  		}
 //    	std::cout<<"\n main "<<std::endl;
 //  	}
	Net<float>* elm = new Net<float>(network,TRAIN);
	cout<<"Network Initialised"<<endl;
	dnn_fwd(a,b,elm);
	cout<<"Forward Finished"<<endl;
	dnn_fwd(a,b,elm);
	cout<<"Forward Finished"<<endl;
	dnn_fwd(a,b,elm);
	cout<<"Forward Finished"<<endl;
	dnn_fwd(a,b,elm);
	cout<<"Forward Finished"<<endl;
	// dnn_fwd(a,b,elm,elm_output);
	// cout<<"Forward Finished"<<endl;
	// dnn_fwd(a,b,elm,elm_output);
	// cout<<"Forward Finished"<<endl;
	// print_output(&elm_output,800,8);
	// print_output(&ls_weights,2048,8);
	free(elm);



}
