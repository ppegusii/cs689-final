#include "mex.h"
#include <math.h>
#include <string.h>

#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void logmtimes(double *inData1, double *inData2, int Q, int T1, int T2, int N, double *outData);


/* Called with Transmat and logLocEv*/
void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	 int i, j, Q, Q1, Q2, T, N;
	 double *TransMat,*logLocalEv, *belData, *belief, *fwMsg, *tempFwMsg;
	 mxArray *tempDoubleMat, *tempDoubleMat2;
	 if ((nlhs> 2) | (nrhs != 2))
	{
		mexErrMsgTxt("The number of input should be 2 and output arguments should be 2");
	}
	 /* Matrix multiplication: QxT * TxN = QxN*/
	 Q1 = mxGetM(prhs[0]);
	 Q2 = mxGetN(prhs[0]);
	 Q = mxGetM(prhs[1]);
	 T = mxGetN(prhs[1]);

	if ((Q1!=Q2) | (Q!=Q1))
	{
		mexErrMsgTxt("Error matrix dimensions of Q don't match.");
	}
	/* Output */
	plhs[0] = mxCreateDoubleMatrix(Q, T, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(Q, T, mxREAL);
	belData = mxGetPr(plhs[0]);
	fwMsg = mxGetPr(plhs[1]);

	/* Input */
	TransMat = mxGetPr(prhs[0]);
	logLocalEv = mxGetPr(prhs[1]);

	tempDoubleMat = mxCreateDoubleMatrix(Q,1, mxREAL);
	belief = mxGetPr(tempDoubleMat);

	tempDoubleMat2 = mxCreateDoubleMatrix(Q,1, mxREAL);
	tempFwMsg = mxGetPr(tempDoubleMat2);

	/* First timeslice*/
	memcpy(belData, logLocalEv, Q*sizeof(double));

	/* in matlab for t=2:T*/
	for(i=1;i<T;i++)
	{
		memcpy(belief, belData+Q*(i-1), Q*sizeof(double));
		logmtimes(TransMat, belief, Q, Q, Q, 1, tempFwMsg);
		memcpy(fwMsg+Q*(i-1), tempFwMsg, Q*sizeof(double));
		for(j=0;j<Q;j++)
		{
			belData[Q*i+j] = tempFwMsg[j] + logLocalEv[Q*i+j];
		}
	}
}
         


void logmtimes(double *inData1, double *inData2, int Q, int T1, int T2, int N, double *outData)
{
	 int i, j, T;
	 double *tempMultMat, *maxVal, tempVal;
	 mxArray *tempDoubleMat, *maxDoubleMat;

	if (T1!=T2)
	{
		mexErrMsgTxt("Matrix multiplication: QxT * TxN = QxN. Error matrix dimensions of T don't match.");
	}
	T = T1;

	tempDoubleMat = mxCreateDoubleMatrix(Q*N,T, mxREAL);
	tempMultMat = mxGetPr(tempDoubleMat);

	maxDoubleMat = mxCreateDoubleMatrix(Q*N,1, mxREAL);
	maxVal = mxGetPr(maxDoubleMat);

	/* Multiply values (in log space this becomes sum)*/
	for (i = 0; i < Q*N; i++)
	{
		for (j = 0; j < T; j++)
		{
			tempMultMat[(i*T)+j] = inData1[j*Q+(i/N)] + inData2[(i%N)*T+j];

			if (j==0)
			{
				maxVal[i] = tempMultMat[(i*T)+j];
			}
			else
			{
				maxVal[i] = max(maxVal[i],tempMultMat[(i*T)+j]);
			}			
		}
	}

	/* Sum values (in log space this becomes logsum)*/
	for (i = 0; i < Q*N; i++)
	{
		tempVal = 0;
		for (j = 0; j < T; j++)
		{
			tempVal +=exp(tempMultMat[(i*T)+j] - maxVal[i]);
		}
		outData[((i%N)*Q)+(i/N)] = log(tempVal) + maxVal[i];
	}
}
             
