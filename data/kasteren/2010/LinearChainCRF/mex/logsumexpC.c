#include "mex.h"
#include <math.h>
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
 int i, j, m, n, d;
 double *inData,*outData, maxVal;
 if (nlhs!= 1 | nrhs != 2)
{
	mexErrMsgTxt("The number of input should be 2 and output arguments should be 1");
}
 
 d = 1;/*(mxGetScalar(prhs[1]);*/
 m = mxGetM(prhs[0]);
 n = mxGetN(prhs[0]);
 
 if (d==1)
	 {
 	plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
	 }
 else
	 {
 	plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
	 }
 inData = mxGetPr(prhs[0]);
 outData = mxGetPr(plhs[0]);
	 
 maxVal =inData[0];
 
for (j = 0; j < m*n; j++)
{
	if (max(maxVal,inData[j]))
		maxVal = inData[j];
}

for (i = 0; i < n; i++)
{
	for (j = 0; j < m; j++)
	{
		outData[i] += exp(inData[i*m+j] - maxVal);
	}
	outData[i] = log(outData[i]) + maxVal;
}
}
             
