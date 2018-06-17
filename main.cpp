#include "stdafx.h"
#include <iostream>
#include "Rnn.h"

int main()
{
	srand(time(NULL));
	
	Rnn rnn;

	rnn.train();
		
	return 0;
}
