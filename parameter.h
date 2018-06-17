#include <cmath>
#include "stdafx.h"
#include <math.h>

const int binary_dim = 8;
const int largest_number = pow(2, binary_dim);

#define randval(high) ( (double)rand() / RAND_MAX * high )
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 )
const int input_node = 2;
const int hidden_node = 16;
const int output_node = 1;
const double alpha = 0.1;