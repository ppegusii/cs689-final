#include "mex.h"
#include <math.h>
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	 int i, j, m, n, d;
	 double *inData,*outData, maxVal, tempVal;
	 if (nlhs> 1 | nrhs != 1)
	{
		mexErrMsgTxt("The number of input should be 1 and output arguments should be 1");
	}
	 m = mxGetM(prhs[0]);
	 n = mxGetN(prhs[0]);
	 

	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

	inData = mxGetPr(prhs[0]);
	outData = mxGetPr(plhs[0]);
		 
	maxVal =inData[0];
	 
	for (j = 0; j < m*n; j++)
	{
		maxVal = max(maxVal,inData[j]);
	}

	tempVal = 0;
	for (j = 0; j < m*n; j++)
	{
		tempVal +=exp(inData[j] - maxVal);
	}
	outData[0]=log(tempVal) + maxVal;
}
             
