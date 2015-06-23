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


#define FEATURE_VEC_SIZE 65  // Number of floats in one input feature vector
#define PROB_VEC_SIZE 8  // Number of floats in one output probability vector

// void load_features(float** in,std::string filename){
// 	float data[200][65];
// 	*in = (float*)malloc(sizeof(float) * 200 * 8);
// 	ifstream file(filename);
// 	for( int row = 0 ; row < 200 ; row++ ){
// 		string line;
// 		getline(file, line);
// 		if( !file.good() )
// 			break;
// 		stringstream iss(line);
//
// 		for( int col = 0 ; col < 65 ; col++ ){
// 			string val;
// 			getline( iss, val, ' ');
// 			if( !iss.good() )
// 				break;
// 			stringstream convertor(val);
// 			convertor >> data[row][col];
// 			(*in)[row*65 + col] = data[row][col];
// 		}
// 	}
//
// 	for( int row = 0 ; row < 200 ; row++){
// 		for( int col = 0 ; col < 65 ; col++){
// 			cout << "i : "<<row<<" j : "<<col<<"  "<<(*in)[row*65 + col]<<endl;
// 			cout << " , ";
//
// 		}
// 		cout << endl;
// 	}
//
// }

// void load_labels(float** in,string filename){
// 	int labels[200];
// 	*in = (float*)malloc(sizeof(float) * 200 * 8);
// 	ifstream file(filename);
// 	for( int row = 0 ; row < 200 ; row++ ){
// 		string line;
// 		std::getline(file, line);
// 		if( !file.good() )
// 			break;
// 		stringstream iss(line);
// 		iss >> labels[row];
// 	}
// 	float pro_labels[200][8];
// 	for ( int row = 0 ; row < 200 ; row++ ){
// 		for ( int col = 0 ; col < 8 ; col ++){
// 			if( (int)labels[row] == (col+1) )
// 				pro_labels[row][col] = 1;
// 			else
// 				pro_labels[row][col] = -1;
// 			(*in)[row*8 + col] = pro_labels[row][col];
// 		}
// 	}
//
// 	for( int row = 0 ; row < 200 ; row++){
// 		for( int col = 0 ; col < 8 ; col++){
// 			cout << pro_labels[row][col];
// 			cout << " , ";
// 		}
// 		cout << endl;
// 	}
//
//
// }
inline bool isEqual(float x, float y) {
  const float epsilon = pow(10, -4);
  return abs(x - y) <= epsilon;
}

int load_features(float** in, string feature_file, int vec_size) {
  // Read in features from file
  // First need to detect how many feature vectors
  ifstream inFile(feature_file.c_str(), ifstream::in);
  int feat_cnt = count(istreambuf_iterator<char>(inFile),
                       istreambuf_iterator<char>(), '\n') -
                 1;

  // Allocate memory for input feature array
  *in = (float*)malloc(sizeof(float) * feat_cnt * vec_size);

  // Read the feature in
  int idx = 0;
  ifstream featFile(feature_file.c_str(), ifstream::in);
  string line;
  getline(featFile, line);  // Get rid of the first line
  while (getline(featFile, line)) {
    istringstream iss(line);
    float temp;
    while (iss >> temp) {
      (*in)[idx] = temp;
      idx++;
    }
  }

  // Everything should be in, check for sure
  assert(idx == feat_cnt * vec_size && "Error: Read-in feature not correct.");

  return feat_cnt;
}

void process_label(float* elm_output, float* exp_out){
	int label[800];
	int count=0;
	for (int i = 0; i < 800; i++){
		float max=0;
		for (int j = 0; j < 8; j++){
			if(elm_output[i*8+j]>max){
				max = elm_output[i*8+j];
				label[i]=j+1;
			}

		}
		cout<<label[i]<<endl;
		if(exp_out[i]==(float)label[i])
			count++;
	}
	cout<<count<<endl;

}
void process_label_train(float* labels, float* labels2d){
	for ( int row = 0 ; row < 800 ; row++ ){
		for ( int col = 0 ; col < 8 ; col ++){
			if( (int)labels[row] == (col+1) )
				labels2d[row*8+col] = 1;
			else
				labels2d[row*8+col] = -1;
		}
	}
}
void dnn_fwd(float* in, float* out, Net<float>* net, float* predicted){
	cout << "going to get input blobs" << endl;
	vector<caffe::Blob<float>*> in_blobs = net->input_blobs();
	cout << "got input blobs" << endl;
	vector<caffe::Blob<float>*> out_blobs;
	float loss;
	cout << "going to set data" << endl;
	in_blobs[0]->set_cpu_data(in);
	cout << "setted in_blob[0]" << endl;
	in_blobs[1]->set_cpu_data(out);
	cout << "setted in_blob[1]" << endl;
  out_blobs =	net->ForwardPrefilled(&loss);
	memcpy(predicted,out_blobs[0]->cpu_data(), sizeof(float) *8*800);
  cout << "ForwardPrefilled finished"<< endl;

}
int main(int argc, char** argv){
	Caffe::set_mode(Caffe::CPU);
	string network(argv[1]);
	string features(argv[2]);
	string labels(argv[3]);
	float* input = NULL;
	float* exp_out1 = NULL;
	float* elm_output = (float*)malloc(sizeof(float)*8*800);
	// load_features(&input, features);
	int feat_cnt = load_features(&input, features, FEATURE_VEC_SIZE);

	// load_labels(&exp_out, labels);
	int correct_out_cnt = load_features(&exp_out1, labels, PROB_VEC_SIZE);
	float* exp_out = (float*)malloc(sizeof(float)*8*800);
	process_label_train(exp_out1,exp_out);
	Net<float>* elm = new Net<float>(network,TRAIN);
	cout<<"Network Initialised"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	dnn_fwd(input,exp_out,elm,elm_output);
	cout<<"Finished"<<endl;
	float sqerror = 0;
	for (int i = 0; i < 800; i++){
		for(int j=0; j < 8; j++){
			std::cout<<" main: "<<elm_output[i*8+j];
		}
		std::cout<<"\n"<<std::endl;
	}
	for (int i = 0; i < (8*800); i++)
    sqerror += (exp_out[i]-elm_output[i])*(exp_out[i]-elm_output[i]);
	cout<<sqerror<<endl;
	float mse = sqerror/(8*800);
	cout<<mse<<endl;
	process_label(elm_output,exp_out1);
  cout<<"training successfull"<<endl;
  /////// Testing

  cout<<"Testing net"<<endl;
  Net<float>* elm_t = new Net<float>(network,TEST);
  elm_t->ShareTrainedLayersWith(elm);
	dnn_fwd(input,exp_out,elm_t,elm_output);
	process_label(elm_output,exp_out1);
  cout<<"sharing successfull"<<endl;
}
