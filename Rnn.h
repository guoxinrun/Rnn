#include "stdafx.h"
#include <iostream>
#include <string>
#include <cmath>
#include <math.h>
#include <vector>
#include "time.h"
#include"parameter.h"
#include "test.h"


//Use the Rnn model to predict the result of binary addition
class Rnn{

	friend 

public:

	Rnn();
	void train();
	virtual ~Rnn();

private:

	double weight_of_in_hidden[input_node][hidden_node];
	double weight_of_out_hidden[hidden_node][output_node];
	double weight_of_pre_cur_hidden[hidden_node][hidden_node];

	std::vector<double>* layer_input;
	std::vector<double>* layer_output;
	
	double sigmoid(const double& );
	double sigmoid_output_to_derivative(const double& );
	std::vector<int> int_to_binary(const int& );
	void cout_array(const std::vector<int>& );
	void weight_init(double* , const int& );
		void print_predict_err(const double&, const int*,
		const std::vector<int>&, const int&, const int&);
	void clear_layer(std::vector<double*>&, std::vector<double>&);
	void init_val(int&, int&, int&, std::vector<int>&, std::vector<int>&, std::vector<int>&);
	void forw_input_to_hidden(const std::vector<double>*, const std::vector<double*>&, double*);
	void forw_hidden_to_out(const double*, std::vector<double>*);
	void update_weight_of_out_hidden(const double*, const std::vector<double>&, const int);
	void update_other_weight(double*, const std::vector<double>&, const double*, const int, const double*, const double*);
	
};

//Define the sigmoid funtion
double Rnn::sigmoid(const double& input){
	double output = 1.0 / (1.0 + exp(-input));
	return output;
}

//Define the derivative of sigmoid funtion
double Rnn::sigmoid_output_to_derivative(const double& input){
	double output = input*(1 - input);
	return output;
}


//Convert the int number to binary form
std::vector<int> Rnn::int_to_binary(const int& largest_num){

	std::vector<int> binary_array(binary_dim);
	int largest_num_copy = largest_num;
	int count = 0;
	while (largest_num_copy){
		binary_array[count++] = largest_num_copy % 2;
		largest_num_copy /= 2;
	}
	while (count<binary_dim){
		binary_array[count++] = 0;
	}
	return binary_array;
}
typedef std::vector<int> Path;

//Print the element of vector
void Rnn::cout_array(const std::vector<int>& array){
	if (array.size() == 0) {
		throw  std::exception("cout array is NULL");
		return;
	}

	for (Path::size_type i = 0; i<array.size(); ++i){
		std::cout << array[i] << " ";
	}
}

//Initialize the weight
void Rnn::weight_init(double weight[], const int& size){
	if (weight == NULL) {
		throw new std::exception("The array is empty");
		return;
	}

	for (int i = 0; i<size; i++){

		weight[i] = uniform_plus_minus_one;
	}
}

Rnn::Rnn(){

	weight_init((double*)weight_of_in_hidden, input_node*hidden_node);
	weight_init((double*)weight_of_out_hidden, output_node*hidden_node);
	weight_init((double*)weight_of_pre_cur_hidden, hidden_node*hidden_node);

	layer_input = new std::vector<double>(input_node);
	layer_output = new std::vector<double>(output_node);
}

Rnn::~Rnn(){
	delete layer_input;
	delete layer_output;
}

//Print the current error of data predicted
void Rnn::print_predict_err(const double& err, const int* pred_res,
	const std::vector<int>& true_res, const int& input_one,
	const int& input_two){
	std::cout << "error: " << err << std::endl;
	std::cout << "predict: ";
	for (int i = 0; i<binary_dim; i++){
		std::cout << pred_res[i];
	}
	std::cout << std::endl;

	std::cout << "true: ";
	for (int j = 0; j<true_res.size(); j++){
		std::cout << true_res[j];
	}
	std::cout << std::endl;

	int output = 0;
	for (int k = 0; k<binary_dim; k++){
		output += pred_res[k] * pow(2, k);
	}

	std::cout << input_one << " + " << input_two << " = " << output << std::endl << std::endl;
}

void Rnn::clear_layer(std::vector<double*>& layer_hid, std::vector<double>& layer_out_del){
	for (int i = 0; i<layer_hid.size(); i++){
		delete layer_hid[i];
	}
	layer_hid.clear();
	layer_out_del.clear();
}

//Initialize the input and output of binary addition
void Rnn::init_val(int& in_one, int& in_two, int& out,
	std::vector<int>& in_one_vec, std::vector<int>& in_two_vec,
	std::vector<int>& out_vec){
	in_one = (int)randval(largest_number / 2.0);
	in_two = (int)randval(largest_number / 2.0);
	out = in_one + in_two;
	in_one_vec = int_to_binary(in_one);
	in_two_vec = int_to_binary(in_two);
	out_vec = int_to_binary(out);

}


//Forward propagation procession from input layer to hidden layer
void Rnn::forw_input_to_hidden(const std::vector<double>*layer_input_temp,
	const std::vector<double*>& layer_hidden_p_vec_temp, double* middle_res){
	for (int i = 0; i<hidden_node; i++){
		double res_one = 0.0;
		//input layer to hidden layer
		for (int m = 0; m<input_node; m++){
			res_one += (*layer_input_temp)[m] * weight_of_in_hidden[m][i];
		}

		//previous hidden layer to current hidden layer
		double* layer_pre_hidden = layer_hidden_p_vec_temp.back();
		for (int k = 0; k<hidden_node; k++){
			res_one += layer_pre_hidden[k] * weight_of_pre_cur_hidden[k][i];
		}

		middle_res[i] = sigmoid(res_one);

	}
}

//Forward propagation procession from hidden layer to output layer
void Rnn::forw_hidden_to_out(const double* layer_hid_p, std::vector<double>* layer_out){
	for (int i = 0; i<output_node; i++){
		double output = 0.0;
		for (int j = 0; j<hidden_node; j++){
			output += layer_hid_p[j] * weight_of_out_hidden[j][i];

		}
		(*layer_out)[i] = sigmoid(output);
	}
}

//Backward propagation procession to update the weight between out layer and hidden layer;
void Rnn::update_weight_of_out_hidden(const double* layer_hidden_p_temp,
	const std::vector<double>& layer_loss_delta_out_temp, const int back_temp){
	for (int i = 0; i<output_node; i++){
		for (int j = 0; j<hidden_node; j++){
			weight_of_out_hidden[j][i] += alpha*layer_hidden_p_temp[j] * layer_loss_delta_out_temp[back_temp];

		}
	}
}

//Backward propagation procession to update other two weight 
void Rnn::update_other_weight(double* out_hid_delta,
	const std::vector<double>& layer_loss_delta_out_temp, const double* later_cur_hid_delta, const int back_temp,
	const double* layer_hidden_p_tem, const double* layer_pre_hidden_p_temp){
	int j;
	for (int i = 0; i<hidden_node; i++){
		out_hid_delta[i] = 0;

		for (j = 0; j<output_node; j++){
			out_hid_delta[i] += layer_loss_delta_out_temp[back_temp] * weight_of_out_hidden[i][j];
		}
		for (j = 0; j<hidden_node; j++){
			out_hid_delta[i] += later_cur_hid_delta[j] * weight_of_pre_cur_hidden[i][j];
		}

		out_hid_delta[i] = out_hid_delta[i] * sigmoid_output_to_derivative(layer_hidden_p_tem[i]);

		for (j = 0; j<input_node; j++){
			weight_of_in_hidden[j][i] += alpha*out_hid_delta[i] * (*layer_input)[j];

		}

		for (j = 0; j<hidden_node; j++){
			weight_of_pre_cur_hidden[j][i] += alpha*out_hid_delta[i] * (layer_pre_hidden_p_temp)[j];

		}

	}


}


void Rnn::train(){
	std::vector<double*> layer_hidden_p_vec;
	std::vector<double> layer_loss_delta_out;

	for (int epoch = 0; epoch<10000; epoch++){
		double error = 0.0;
		int i, j;

		for (i = 0; i<layer_hidden_p_vec.size(); i++)
			delete layer_hidden_p_vec[i];
		layer_hidden_p_vec.clear();
		layer_loss_delta_out.clear();


		int pred[binary_dim];
		memset(pred, 0, sizeof(pred));

		//Initialize the input and result vector
		int add_one, add_two, true_res;
		std::vector<int> add_one_vector(binary_dim);
		std::vector<int> add_two_vector(binary_dim);
		std::vector<int> true_res_vector(binary_dim);
		std::vector<int> pred_res_vector(binary_dim);
		init_val(add_one, add_two, true_res, add_one_vector, add_two_vector, true_res_vector);

		double* layer_hidden_p = new double[hidden_node];


		for (i = 0; i<hidden_node; i++)
			layer_hidden_p[i] = 0;								//Initialize the foremost hidden layer by 0
		layer_hidden_p_vec.push_back(layer_hidden_p);


		//forward propagation 
		for (int forw = 0; forw<binary_dim; forw++){

			(*layer_input)[0] = add_one_vector[forw];
			(*layer_input)[1] = add_two_vector[forw];
			double y = (double)true_res_vector[forw];
			layer_hidden_p = new double[hidden_node];


			forw_input_to_hidden(layer_input, layer_hidden_p_vec, layer_hidden_p);
			forw_hidden_to_out(layer_hidden_p, layer_output);


			pred[forw] = (int)floor((*layer_output)[0] + 0.5);
			layer_hidden_p_vec.push_back(layer_hidden_p);

			layer_loss_delta_out.push_back((y - (*layer_output)[0])*sigmoid_output_to_derivative((*layer_output)[0]));
			error += fabs(y - (*layer_output)[0]);
		}

		//backward propagation
		double* out_hid_delta = new double[hidden_node];
		double* later_cur_hid_delta = new double[hidden_node];
		for (i = 0; i<hidden_node; i++){
			later_cur_hid_delta[i] = 0;
		}
		for (int back = binary_dim - 1; back >= 0; back--){
			(*layer_input)[0] = add_one_vector[back];
			(*layer_input)[1] = add_two_vector[back];

			layer_hidden_p = layer_hidden_p_vec[back + 1];
			double* layer_pre_hidden_p = layer_hidden_p_vec[back];

			update_weight_of_out_hidden(layer_hidden_p, layer_loss_delta_out, back);
			update_other_weight(out_hid_delta, layer_loss_delta_out, later_cur_hid_delta,
				back, layer_hidden_p, layer_pre_hidden_p);

			if (back == binary_dim - 1)
				delete later_cur_hid_delta;
			later_cur_hid_delta = out_hid_delta;

		}

		delete later_cur_hid_delta;

		//Print the error
		if (epoch % 1000 == 0)
			print_predict_err(error, pred, true_res_vector, add_one, add_two);
	}
}

