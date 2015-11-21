#include "mex.h"
#include <math.h>
#include <string.h>
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void logmtimes(double *inData1, double *inData2, int Q, int T1, int T2, int N, double *outData);
void logSumWithMax(double *inData, double maxVal, int m, int n, double *outData );
void calcTransExp(double *belInData, double *tempBel, double *TransMat, int Q, double *belTransData);

/*In Matlab: [bel,belTrans] = crfBkProbC(bel, fwdMsg, TransMat');*/
void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	 int i, j, Q, Q1, Q2, T, N;
	 double *TransMat,*belTransData, *belInData, *belOutData, *belief, *fwMsg, *tempBkMsg, *tempBel;
	 mxArray *tempDoubleMat, *tempDoubleMat2, *tempDoubleMat3;
	 if ((nlhs> 2) | (nrhs != 3))
	{
		mexErrMsgTxt("The number of input should be 2 and output arguments should be 2");
	}
	 Q = mxGetM(prhs[0]);
	 T = mxGetN(prhs[0]);

	/* Output */
	plhs[0] = mxCreateDoubleMatrix(Q, T, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(Q, Q, mxREAL);
	belOutData = mxGetPr(plhs[0]);
	belTransData = mxGetPr(plhs[1]);

	/* Input */
	belInData = mxGetPr(prhs[0]);
	fwMsg = mxGetPr(prhs[1]);
	TransMat = mxGetPr(prhs[2]);

	tempDoubleMat = mxCreateDoubleMatrix(Q,1, mxREAL);
	belief = mxGetPr(tempDoubleMat);

	tempDoubleMat2 = mxCreateDoubleMatrix(Q,1, mxREAL);
	tempBkMsg = mxGetPr(tempDoubleMat2);

	tempDoubleMat3 = mxCreateDoubleMatrix(Q,1, mxREAL);
	tempBel = mxGetPr(tempDoubleMat3);

	/* Last timeslice */
	memcpy(belOutData+Q*(T-1), belInData+Q*(T-1), Q*sizeof(double));

	/* in matlab for t=T-1:-1:1 */
	for(i=T-2;i>=0;i--)
	{
		for(j=0;j<Q;j++)
		{
			tempBel[j] = belOutData[Q*(i+1)+j] - fwMsg[Q*i+j];
		}
		calcTransExp(belInData+(Q*i),tempBel,TransMat,Q,belTransData);

		logmtimes(TransMat, tempBel, Q, Q, Q, 1, tempBkMsg);
		for(j=0;j<Q;j++)
		{
			belOutData[Q*i+j] = belInData[Q*i+j] + tempBkMsg[j];
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

	/* Multiply values (in log space this becomes sum) */
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

	/* Sum values (in log space this becomes logsum) */
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
      

/*In Matlab: belTrans(:,:,t) = crfTransExpC(bel(:,t), tmp(:)', TransMat); */
void calcTransExp(double *belInData, double *tempBel, double *TransMat, int Q, double *belTransData)
{
	 int i, j;
	 double *belttp1, *transTotal, maxVal;
	 mxArray *tempDoubleMat, *tempDoubleMat2;

	tempDoubleMat = mxCreateDoubleMatrix(Q,Q, mxREAL);
	belttp1 = mxGetPr(tempDoubleMat);

	tempDoubleMat2 = mxCreateDoubleMatrix(1,1, mxREAL);
	transTotal = mxGetPr(tempDoubleMat2);

	maxVal = belInData[0] + tempBel[0] + TransMat[0];
	for(i=0;i<Q;i++)
	{
		for(j=0;j<Q;j++)
		{
			belttp1[j*Q+i] = belInData[i] + tempBel[j] + TransMat[j*Q+i];
			maxVal = max(maxVal, belttp1[i*Q+j]);
		}
	}
	logSumWithMax(belttp1, maxVal, Q, Q, transTotal);

	for(i=0;i<Q;i++)
	{
		for(j=0;j<Q;j++)
		{
			belTransData[i*Q+j] = belTransData[i*Q+j] + exp(belttp1[i*Q+j] - *transTotal);
		}
	}
}


void logSumWithMax(double *inData, double maxVal, int m, int n, double *outData)
{
	 int i, j;
	 double tempVal;

	tempVal = 0;
	for (j = 0; j < m*n; j++)
	{
		tempVal +=exp(inData[j] - maxVal);
	}
	outData[0]=log(tempVal) + maxVal;
}
             
