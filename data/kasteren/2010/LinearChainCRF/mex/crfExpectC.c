#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
 int i, j, m, n, t, T, Q, O, numLambda;
 double *data1, *data2;
 if (nlhs!= 2 | nrhs != 3)
{
	mexErrMsgTxt("The number of input should be 3 and output arguments should be 2");
}
 plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
 plhs[1] = mxCreateDoubleMatrix(mxGetM(prhs[0]),1, mxREAL);

 T = mxGetN(prhs[1]);
 O = mxGetM(prhs[1]);
 Q = mxGetM(prhs[2]);
 numLambda = mxGetN(prhs[0]);



 for (t = 1; t < T; t++) 
 {
	
 }

 
/*
 for (i = 0; i < nrhs; i++) 
   {
    m = mxGetM(prhs[i]);
    n = mxGetN(prhs[i]);
 

    plhs[i] = mxCreateDoubleMatrix(m, n, mxREAL);


    data1 = mxGetPr(prhs[i]);


    data2 = mxGetPr(plhs[i]);


    for (j = 0; j < m*n; j++)
    {
    data2[j] = 2 * data1[j];
	mexPrintf("%f",data2[j]);
    }
   }   
*/
}
             