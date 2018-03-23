// ------------------头文件------------------
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cufft.h"
#include "cufftXt.h"
#include "cufftw.h"
#include<complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <memory.h>
#include<malloc.h>
#include<iostream>
using namespace std;
#include <windows.h>
#include <process.h>

//131072点原始数据滤波得到65536点滤波后通道数据，50%重叠
//65536点滤波后通道数据降采样得到16384点通道数据
//做16384点频域波束形成，找目标方向
//做32768点时域波束形成，得到16384点跟踪波束
//时域降采样后的数据累积长度为32768点，50%重叠

// -----------------宏定义-----------------------
//#define		ARNUM                    96       // 阵元个数 
//#define     SAMPNUM                  16384    // 波束形成帧长
//#define     BeamformNumber           97       // 波束数
#define     NFFT					 16384	  // FFT的点数
#define     PI				         3.1415926f
#define     UWC						 1500.0f     //声速
#define     FS						 100000    // 采样频率
typedef     float2					 Complex;
//#define     blockNum                 97
#define     threadsPerBlock          512
#define     d                        0.07f
#define     FL                       100.0f
#define     FH                       4000.0f
#define     TL                       17
#define     CHANNUM                  16
#define     FRAMELEN                 65536
#define     DOWNSAMPLE               4
#define     FIRORDER                 2048
#define     FILTER_FRAME             (2*FRAMELEN)
#define     BEAMNUM                  91
#define     THREADNUMPERBLK          256
#define     ARRAYNUM                 15
#define     STARTBEAM                15
#define     ENDBEAM                  75
#define     MAXTRACETARNUM           3
#define     M                        3
#define     PSD_LEN                  20
#define     PSD_AVG_NUM              8
#define     EPS                      1e-8
#define     SMOOTH_N                 100
#define     LINE_NUM                 16
#define     DEM_RST_LEN              1024
#define     VECTOR_P_IDX             22
#define     VECTOR_X_IDX             16
#define     VECTOR_Y_IDX             18
// --------------函数声明--------------------------
void Run(LPVOID lParam);
void ReadBoard1Data(LPVOID lParam);
void ReadBoard2Data(LPVOID lParam);
void DataFormatting(LPVOID lParam);
void ArraySignalProcessing(LPVOID lParam);
//---------------事件有关的变量--------------------
HANDLE g_hReadBoard1ThreadReadyEnvent;
HANDLE g_hReadBoard2ThreadReadyEnvent;
HANDLE g_hFrameDataReadyEnvent;
//---------------读取数据变量---------------------
int *DataBufA_B1 = NULL;
int *DataBufB_B1 = NULL;
int *DataBufA_B2 = NULL;
int *DataBufB_B2 = NULL;
float *ChannDataBufA = NULL;
float *ChannDataBufB = NULL;
float *DownSamplingDataBufA = NULL;
float *DownSamplingDataBufB = NULL;
//------------------滤波降采样变量-----------------
int fir1(int n,int band,float fl,float fh,float fs,int wn, float *h);
float window(int type,int n,int i,float beta);
float kaiser(int i,int n,float beta);
float bessel0(float x);
void findpeak(float *data, int *p,int dn);
void findvalley(float *data, int *p,int dn);
bool peakdetection(int beamidx,float *be,int *valley,float threshold);
void rbub(float *p,int *idx,int n);
void MySmooth(float *datain,int nDataLen,float *paraA,int nParaLen,int nOrder,int nWindow,int nStep,float *dataout);
void CalSmoothPara(float *para);

//功率谱分析
float fSmoothA[4][SMOOTH_N]={0.0};                //滑动窗正交多项式拟合时所用规范正交向量
float fPlineInfo[MAXTRACETARNUM][LINE_NUM][4]={0};//功率谱信息
float fDlineInfo[MAXTRACETARNUM][LINE_NUM][2]={0};//解调谱信息
int   nPlineNum = 0;
int   nDlineNum = 0;
int   nVectorPlineNum = 0;
float fVectorPlineInfo[LINE_NUM][4]={0};          //功率谱信息
//解调谱分析
int   DemFreqBandNum=0;                           //解调谱分析分频带数，默认最多分10个频带
float DemStartFreq[10]={0.0};                     //解调谱分析分频带起始频率
float DemEndFreq[10]={0.0};                       //解调谱分析分频带结束频率
// -----------------主函数-------------------------
int main(int argc, char **argv)
{
	 g_hReadBoard1ThreadReadyEnvent = CreateEvent(NULL,FALSE,FALSE,NULL);
	 g_hReadBoard2ThreadReadyEnvent = CreateEvent(NULL,FALSE,FALSE,NULL);
	 g_hFrameDataReadyEnvent = CreateEvent(NULL,FALSE,FALSE,NULL);

	 //_beginthread(Run,0,NULL);	
	 _beginthread(ArraySignalProcessing,0,NULL);
	 _beginthread(DataFormatting,0,NULL);
	 _beginthread(ReadBoard1Data,0,NULL);
	 _beginthread(ReadBoard2Data,0,NULL);

     Sleep(1000);
	 getchar();
	 getchar();
     return 0;
}

__global__ void PhiShiftFactorGen(cufftComplex *XNSS)
{
	int bid = 0,tid = 0;
	float tt = 0.0f;
	float angle=0.0f;
	float det[ARRAYNUM];
	float MovePoints[ARRAYNUM];

	bid = blockIdx.x;
	tid = threadIdx.x;
	angle=float(tid*PI/(BEAMNUM-1));

	for(int i=0;i<ARRAYNUM;i++)
	{		
	   det[i]=i*d*cos(angle)/UWC;
	   MovePoints[i]=det[i]*FS/DOWNSAMPLE;
	   tt=MovePoints[i]*2*PI*bid/NFFT;
	   XNSS[tid*ARRAYNUM*NFFT/2+i*NFFT/2+bid].x = cos(tt);
	   XNSS[tid*ARRAYNUM*NFFT/2+i*NFFT/2+bid].y = sin(tt);
	}
}

//__global__ void f_beamform(cufftComplex *dev_fft,cufftReal *dev_energy,int *FreqBin,cufftComplex *PhiArray)
//{
//	__shared__ float Mabs[NFFT/2];
//	float tempX=0.0f;
//	float tempY=0.0f;
//	int nfl = 0;
//	int nfh = 0;
//	int freqbinnum = 0;
//	cuComplex XNSS;
//	cuComplex XFFTafterPinYi;
//	float ax = 0.0f,ay=0.0f,bx=0.0f,by=0.0f;
//	float energyEachBoShu = 0.0f;
//	int bid = 0,tid = 0;
//
//	nfl = (int)((FL/FS*NFFT)+0.5);
//	nfh = (int)((FH/FS*NFFT)+0.5);
//	freqbinnum = *(FreqBin);
//	bid = blockIdx.x;
//	tid = threadIdx.x;
//
//	if(tid==0)
//	{
//		memset(Mabs,0,sizeof(float)*NFFT/2);
//	}
//    // -----------------每个线程对应频点能量和----------------------
//	//每个线程计算freqbinnum个频点的能量和
//	for(int j=0;j<freqbinnum;j++)
//	{
//		tempX=0.0;
//		tempY=0.0;
//		for(int i=0;i<ARNUM;i++)
//		{		
//		   XNSS.x=PhiArray[bid*ARNUM*(NFFT/2)+i*(NFFT/2)+tid*freqbinnum+j].x;
//		   XNSS.y=PhiArray[bid*ARNUM*(NFFT/2)+i*(NFFT/2)+tid*freqbinnum+j].y;
//		   ax=dev_fft[i*(NFFT/2)+(tid*freqbinnum+j)].x;
//		   ay=dev_fft[i*(NFFT/2)+(tid*freqbinnum+j)].y;
//		   bx=XNSS.x;
//		   by=XNSS.y;
//
//		   if (tid*freqbinnum+j>= nfl && tid*freqbinnum+j<=nfh)
//		   {
//				XFFTafterPinYi.x=ax*bx-ay*by;
//				XFFTafterPinYi.y=ax*by+bx*ay;
//		   }
//		   else
//		   {
//				XFFTafterPinYi.x=0;
//				XFFTafterPinYi.y=0;
//		   }
//
//		   tempX=tempX+ XFFTafterPinYi.x; 
//		   tempY=tempY+ XFFTafterPinYi.y;
//		}
//	
//		Mabs[tid*freqbinnum+j]=pow(tempX,2)+pow(tempY,2);
//
//		//块内线程同步
//		__syncthreads();
//	
//		//-----------------所有频点的能量和相加的运算放在每个块的第一个线程-----------------	
//		if(tid==0)
//		{
//		   energyEachBoShu=0.0f;
//		   for(int k=0;k<NFFT/2;k++)
//		   {
//			   energyEachBoShu=energyEachBoShu+Mabs[k];
//		   }
//		   dev_energy[bid]= energyEachBoShu;	   
//		}	
//	}
//}

//__global__ void frequency_domain_beamform(cufftComplex *dev_fft,cufftReal *dev_energy,cufftComplex *PhiArray)
//{
//	__shared__ float Mabs[THREADNUMPERBLK];
//	float tempX=0.0f;
//	float tempY=0.0f;
//	int nfl = 0;
//	int nfh = 0;
//	cuComplex XNSS;
//	cuComplex XFFTafterPinYi;
//	float ax = 0.0f,ay=0.0f,bx=0.0f,by=0.0f;
//	float energyEachBoShu = 0.0f;
//	int bid = 0,tid = 0;
//	int beamidx = 0, freqidx = 0;
//
//	nfl = (int)((FL/FS*NFFT)+0.5);
//	nfh = (int)((FH/FS*NFFT)+0.5);
//
//	bid = blockIdx.x;
//	tid = threadIdx.x;
//	beamidx = bid % BeamformNumber;
//	freqidx = bid / BeamformNumber*512+tid;
//
//	if(tid==0)
//	{
//		memset(Mabs,0,sizeof(float)*threadsPerBlock);
//	}
//    // -----------------每个线程对应频点能量和----------------------
//	tempX=0.0;
//	tempY=0.0;
//	for(int i=0;i<ARNUM;i++)
//	{		
//		XNSS.x=PhiArray[beamidx*ARNUM*(NFFT/2)+i*(NFFT/2)+freqidx].x;
//		XNSS.y=PhiArray[beamidx*ARNUM*(NFFT/2)+i*(NFFT/2)+freqidx].y;
//		ax=dev_fft[i*(NFFT/2+1)+freqidx].x;
//		ay=dev_fft[i*(NFFT/2+1)+freqidx].y;
//		bx=XNSS.x;
//		by=XNSS.y;
//
//		if (freqidx>= nfl && freqidx<=nfh)
//		{
//			XFFTafterPinYi.x=ax*bx-ay*by;
//			XFFTafterPinYi.y=ax*by+bx*ay;
//		}
//		else
//		{
//			XFFTafterPinYi.x=0;
//			XFFTafterPinYi.y=0;
//		}
//
//		tempX=tempX+ XFFTafterPinYi.x; 
//		tempY=tempY+ XFFTafterPinYi.y;
//	}
//	
//	Mabs[tid]=pow(tempX,2)+pow(tempY,2);
//
//	//块内线程同步
//	__syncthreads();
//	
//	//-----------------所有频点的能量和相加的运算放在每个块的第一个线程-----------------	
//	if(tid==0)
//	{
//		energyEachBoShu=0.0f;
//		for(int k=0;k<threadsPerBlock;k++)
//		{
//			energyEachBoShu=energyEachBoShu+Mabs[k];
//		}
//		dev_energy[bid]= energyEachBoShu;	   
//		//if(bid == 10+97)
//		//{
//		//	printf("dev_energy[%d] = %.3f\n",bid,dev_energy[bid]);
//		//}
//	}	
//
//}

__global__ void FD_Beamform(cufftComplex *dev_fft,cufftReal *dev_energy,cufftComplex *PhiArray,int nfl,int nfh)
{
	__shared__ float Mabs[THREADNUMPERBLK];
	float      tempX=0.0f;
	float      tempY=0.0f;
	cuComplex  XNSS;
	cuComplex  XFFTafterPinYi;
	float      ax = 0.0f,ay=0.0f,bx=0.0f,by=0.0f;
	float      energyEachBoShu = 0.0f;
	int        bid = 0,tid = 0;
	int        beamidx = 0, freqidx = 0;

	bid = blockIdx.x;
	tid = threadIdx.x;
	beamidx = bid % BEAMNUM;
	freqidx = bid / BEAMNUM*THREADNUMPERBLK+tid;

	if(tid==0)
	{
		memset(Mabs,0,sizeof(float)*THREADNUMPERBLK);
	}
	__syncthreads();

    // -----------------每个线程对应频点能量和----------------------
	tempX=0.0;
	tempY=0.0;
	for(int i=0;i<ARRAYNUM;i++)
	{		
		XNSS.x=PhiArray[beamidx*ARRAYNUM*(NFFT/2)+i*(NFFT/2)+freqidx].x;
		XNSS.y=PhiArray[beamidx*ARRAYNUM*(NFFT/2)+i*(NFFT/2)+freqidx].y;
		ax=dev_fft[i*(NFFT/2+1)+freqidx].x;
		ay=dev_fft[i*(NFFT/2+1)+freqidx].y;
		bx=XNSS.x;
		by=XNSS.y;

		if (freqidx>= nfl && freqidx<=nfh)
		{
			XFFTafterPinYi.x=ax*bx-ay*by;
			XFFTafterPinYi.y=ax*by+bx*ay;
		}
		else
		{
			XFFTafterPinYi.x=0;
			XFFTafterPinYi.y=0;
		}

		tempX=tempX+ XFFTafterPinYi.x; 
		tempY=tempY+ XFFTafterPinYi.y;
	}

	Mabs[tid]=pow(tempX,2)+pow(tempY,2);

	//块内线程同步
	__syncthreads();	

	//-----------------所有频点的能量和相加的运算放在每个块的第一个线程-----------------	
	if(tid==0)
	{
		energyEachBoShu=0.0f;
		for(int k=0;k<THREADNUMPERBLK;k++)
		{
			energyEachBoShu=energyEachBoShu+Mabs[k];
		}
		dev_energy[bid]= energyEachBoShu;	   
	}
}

__global__ void MatrixSumRow(cufftReal *dev_energy,cufftReal *sum_energy,int nrow,int ncol)
{
	int bid = 0,tid = 0;
	int row = 0,col = 0;
	float sum = 0.0;
	bid = blockIdx.x;
	row = nrow;
	col = ncol;

	for(int ii = 0;ii<row;ii++)
	{
		sum = sum+dev_energy[ii*col+bid];
	}
	sum_energy[bid] = sum;
}

__global__ void DownSamplingFilter(cufftComplex *dev_fft_sig,cufftComplex *dev_fft_filter,cufftComplex *dev_fft_yk,int FFTN)
{
	int bid = 0,tid = 0;
	cuComplex Sigk;
	cuComplex Hk;
	int chanIdx = 0;
	int freqIdx = 0;

	bid = blockIdx.x;
	tid = threadIdx.x;
	chanIdx = bid % (CHANNUM*2);
	freqIdx = bid / (CHANNUM*2)*THREADNUMPERBLK+tid;
	//for(int ii=0;ii<FFTN/2+1;ii++)
	//{
	//	Sigk.x = dev_fft_sig[bid*FFTN+ii].x;
	//	Sigk.y = dev_fft_sig[bid*FFTN+ii].y;
	//	Hk.x = dev_fft_filter[ii].x;
	//	Hk.y = dev_fft_filter[ii].y;
	//	dev_fft_yk[bid*FFTN+ii].x = Sigk.x*Hk.x-Sigk.y*Hk.y;
	//	dev_fft_yk[bid*FFTN+ii].y = Sigk.x*Hk.y+Sigk.y*Hk.x;
	//}

	Sigk.x = dev_fft_sig[chanIdx*FFTN+freqIdx].x;
	Sigk.y = dev_fft_sig[chanIdx*FFTN+freqIdx].y;
	Hk.x = dev_fft_filter[freqIdx].x;
	Hk.y = dev_fft_filter[freqIdx].y;
	dev_fft_yk[chanIdx*FFTN+freqIdx].x = Sigk.x*Hk.x-Sigk.y*Hk.y;
	dev_fft_yk[chanIdx*FFTN+freqIdx].y = Sigk.x*Hk.y+Sigk.y*Hk.x;

	if( bid/(CHANNUM*2)>= 255 && tid == THREADNUMPERBLK-1)
	{
		Sigk.x = dev_fft_sig[chanIdx*FFTN+FFTN/2].x;
		Sigk.y = dev_fft_sig[chanIdx*FFTN+FFTN/2].y;
		Hk.x = dev_fft_filter[FFTN/2].x;
		Hk.y = dev_fft_filter[FFTN/2].y;
		dev_fft_yk[chanIdx*FFTN+FFTN/2].x = Sigk.x*Hk.x-Sigk.y*Hk.y;
		dev_fft_yk[chanIdx*FFTN+FFTN/2].y = Sigk.x*Hk.y+Sigk.y*Hk.x;
	}
}

__global__ void IFFTNormalize(cufftReal *dev_fft_yout,cufftReal *dev_databuff,int FFTN)
{
	int bid = 0,tid = 0;
	int chanIdx = 0;
	int timeIdx = 0;

	bid = blockIdx.x;
	tid = threadIdx.x;

	chanIdx = bid % (CHANNUM*2);
	timeIdx = bid / (CHANNUM*2)*THREADNUMPERBLK+tid+FFTN/4;
	
	//if(bid < CHANNUM*2 && tid == 0)
	//{
	//	memcpy(dev_databuff+chanIdx*FFTN/DOWNSAMPLE,dev_databuff+chanIdx*FFTN/DOWNSAMPLE+FFTN/DOWNSAMPLE/2,FFTN/DOWNSAMPLE/2*sizeof(float));
	//}

	if(timeIdx % DOWNSAMPLE == 0)
	{
		dev_databuff[chanIdx*FFTN/DOWNSAMPLE + FFTN/DOWNSAMPLE/2 + (timeIdx-FFTN/4)/DOWNSAMPLE] = dev_fft_yout[chanIdx*FFTN+timeIdx] / FFTN;
	}
}

__global__ void DelayFilterGen(float *h,int m,float theta,float *tau,int *dI)
{
	int bid = 0,tid = 0;
	int k=0;
	float dfs = 0.0;
	int DI = 0;
	__shared__ float sum;

	bid = blockIdx.x;
	tid = threadIdx.x;

	if(tid == 0)
	{
		sum = 0.0;
		dfs = bid*d*cos(theta/180.0*PI)/UWC*(FS/DOWNSAMPLE);
		DI = int(bid*d*cos(theta/180.0*PI)/UWC*(FS/DOWNSAMPLE)+0.5);
		tau[bid] =dfs-DI;
		dI[bid] = DI;
		//printf("bid=%d,m=%d,theta = %.3f,dfs = %.3f,DI = %d\n",bid,m,theta,dfs,DI);
	}

	//块内线程同步
	__syncthreads();

	k = tid-m;
	h[bid*(2*m+1)+tid] = sin(k*1.0*PI-tau[bid]*PI+0.000001)/(k*1.0*PI-tau[bid]*PI+0.000001);

	//块内线程同步
	__syncthreads();

	if(tid == 0)
	{
		for(int k=0;k<2*m+1;k++)
		{
			sum = sum + h[bid*(2*m+1)+k];
		}
	}
	__syncthreads();
	
	h[bid*(2*m+1)+tid] =  h[bid*(2*m+1)+tid]/sum;
}

__global__ void FineDelayFilter(cufftReal *dev_xin,cufftReal *dev_yout,cufftReal *delayfilter,int m)
{
	int bid,tid;
	float x=0.0,h=0.0;
	float sum = 0.0;

	bid = blockIdx.x;
	tid = threadIdx.x;
	__shared__ float y[2*M+1];

	if(tid == 0)
	{
		for(int ii=0;ii<2*m;ii++)
		{
			y[ii] = 0.0;
		}
	}
	
	if(bid-2*m+tid >= 0 && bid-2*m+tid < (FILTER_FRAME/DOWNSAMPLE))
	{
		x = dev_xin[bid-2*m+tid];
	}
	if(2*m-tid >=0)
	{
		h = delayfilter[2*m-tid];
	}
	y[tid] = x*h;

	//if(bid == 24855)
	//{
	//	printf("bid = %d,x=%.8f,h=%.8f,y=%.8f\n",bid,x,h,y);
	//}

	//块内线程同步
	__syncthreads();
	if(tid == 0)
	{
		sum = 0.0;
		for(int jj=0;jj<2*m+1;jj++)
		{
			sum = sum + y[jj];
		}
		dev_yout[bid] = sum;
		//if(bid == 24855)
		//{
		//	printf("bid = %d,dev_yout=%.8f\n",bid,dev_yout[bid]);
		//}
	}
}

//void Run(LPVOID lParam)
//{
//	LARGE_INTEGER nFreq;
//    LARGE_INTEGER nBeginTime;
//    LARGE_INTEGER nEndTime;
//	FILE *fp=NULL,*fpw=NULL;
//	int nfl = (int)((FL/FS*NFFT)+0.5);
//	int nfh = (int)((FH/FS*NFFT)+0.5);
//	int FreqbinPerThread = (int)((nfh-nfl+1)/512.0 + 0.5);
//	int FrameNum = 416;
//    double time;
//	int BlockRowNum = 0;
//
//	// -------------初始化并分配主机内存---------------------
//	float *a=NULL,*signaldata=NULL;
//	Complex *sk=NULL;
//	float *debugvar = NULL;
//
//	a = (float *)malloc( sizeof(float) * SAMPNUM * ARNUM );
//	memset(a,0, sizeof(float) * SAMPNUM * ARNUM );
//
//	signaldata = (float *)malloc( sizeof(float) * SAMPNUM * ARNUM );
//	memset(signaldata,0, sizeof(float) * SAMPNUM * ARNUM );
//
//	sk = (Complex *)malloc( sizeof(Complex) * (NFFT/2+0) * ARNUM );
//	memset(sk,0,sizeof(Complex) * (NFFT/2+0) * ARNUM);
//
//	debugvar = (float *)malloc( sizeof(float) * ARNUM * BeamformNumber *(NFFT/2+0));
//	memset(debugvar,0, sizeof(float) * ARNUM * BeamformNumber *(NFFT/2+0));
//
//	float c[BeamformNumber]={0.0};
//
//	// --------------初始化并分配GPU内存--------------------
//	cudaError cudaStatus;
//	cufftReal *dev_a=NULL;
//	cufftComplex *dev_fft=NULL;
//	cufftReal *dev_energy=NULL;//定义设备上变量
//	cufftReal *sum_energy=NULL;//定义设备上变量
//	cufftComplex *PhiArray = NULL;
//	int MatrixRow= NULL;
//	int MatrixCol= NULL;
//	BlockRowNum = NFFT/2/threadsPerBlock;
//  
//	cudaMalloc((void**)&dev_energy,BeamformNumber*BlockRowNum*sizeof(cufftReal));
//	cudaMalloc((void**)&sum_energy,BeamformNumber*sizeof(cufftReal));
//	cudaMalloc((void**)&PhiArray,ARNUM*BeamformNumber*(NFFT/2)*sizeof(cufftComplex));
//	MatrixRow = BlockRowNum;
//	MatrixCol = BeamformNumber;
//
//	cudaStatus = cudaMalloc((void **)&dev_a, sizeof(cufftReal) * SAMPNUM * ARNUM  );
//	if (cudaStatus != cudaSuccess)
//	{
//		printf (" cudaMalloc Error! \n ");
//	}
//	cudaStatus = cudaMalloc((void **)&dev_fft,  sizeof(cufftComplex) * (NFFT/2+1) * ARNUM );
//	if (cudaStatus != cudaSuccess)
//	{
//		printf (" cudaMalloc Error! \n ");
//	}
//
//	cufftHandle plan;   // 创建句柄
//    cufftPlan1d(&plan, SAMPNUM, CUFFT_R2C, 1);  // 对一维句柄进行赋值
//
//	
//    // ------------循环开始------------
//
//	QueryPerformanceFrequency(&nFreq);
//    
//	fp = fopen("D:\\GPUTest\\testdata.bin","rb");
//	fpw = fopen("D:\\GPUTest\\beamenergy.bin","wb");
//
//	PhiShiftFactorGen<<<NFFT/2,BeamformNumber>>>(PhiArray);
//	for(int ii=0;ii<FrameNum;ii++)
//	{
//		fread(a,sizeof(float),SAMPNUM*ARNUM,fp);
//		for(int jj=0;jj<SAMPNUM;jj++)
//		{
//			for(int kk=0;kk<ARNUM;kk++)
//			{
//				signaldata[kk*SAMPNUM+jj] = a[jj*ARNUM+kk];
//			}
//		}
//		cudaMemcpy(dev_a, signaldata, sizeof(cufftReal)*SAMPNUM*ARNUM, cudaMemcpyHostToDevice);//时域数据拷贝到GPU	
//		QueryPerformanceCounter(&nBeginTime); 
//		// ------------FFT--------------------
//		for (int ll=0; ll<ARNUM; ll++)		
//		{		
//			cufftExecR2C(plan, (cufftReal *)&dev_a[ll*SAMPNUM],(cufftComplex *)&dev_fft[ll*(NFFT/2+1)]);  //正FFT
//		}
//		
//		frequency_domain_beamform<<<BlockRowNum*BeamformNumber,threadsPerBlock>>>(dev_fft,dev_energy,PhiArray);//波束形成
//		MatrixSumRow<<<BeamformNumber,1>>>(dev_energy,sum_energy,MatrixRow,MatrixCol);
//
//		////------------变量从设备 拷贝到 主机显示 ------------
//		cudaMemcpy(c,sum_energy,BeamformNumber*sizeof(float),cudaMemcpyDeviceToHost);
//		QueryPerformanceCounter(&nEndTime);
//		//printf("c[37] = %.3f\n",c[37]);
//
//		fwrite(c,sizeof(float),BeamformNumber,fpw);
//
//		time=(double)(nEndTime.QuadPart-nBeginTime.QuadPart)/(double)nFreq.QuadPart;
//		printf("%f\n",time);
//	}
//	fclose(fp);
//	fclose(fpw);
//	free(a); 
//	free(sk);
//	cudaFree(dev_a);
//	cudaFree(dev_fft);
//	cudaFree(dev_energy);
//	cudaDeviceReset();	
//	cufftDestroy(plan);
//	return;
//}

void ReadBoard1Data(LPVOID lParam)
{
	_Longlong fileindex = 0;
	string FilePath = "D:\\20180201宜昌鱼声阵数据\\20180201\\h\\uwrn\\";
	string FileNamePre = "Board1_ADC_";
	string FileIdx = to_string(fileindex);
	string FileNameSur = ".bin";
	string FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
	int DataFileNum = 18;
	FILE *fp = NULL;
	LARGE_INTEGER nFreq;
    LARGE_INTEGER nBeginTime;
    LARGE_INTEGER nEndTime;
	double dftime = 0.0;
	int readbytes = 0;
	int readbuf[TL*CHANNUM+1];
	int BUF_FLAG=0;
	int *pBuf = NULL;
	int *pCounter = NULL;
	int CounterA = FRAMELEN,CounterB = FRAMELEN;
	int temp = 0;

	QueryPerformanceFrequency(&nFreq);

	if(DataBufA_B1 != NULL)
	{
		free(DataBufA_B1);
		DataBufA_B1 = NULL;
	}
	DataBufA_B1 = (int *)malloc(FRAMELEN*CHANNUM*sizeof(int));
	memset(DataBufA_B1,0,FRAMELEN*CHANNUM*sizeof(int));

	if(DataBufB_B1 != NULL)
	{
		free(DataBufB_B1);
		DataBufB_B1 = NULL;
	}
	DataBufB_B1 = (int *)malloc(FRAMELEN*CHANNUM*sizeof(int));
	memset(DataBufB_B1,0,FRAMELEN*CHANNUM*sizeof(int));

	QueryPerformanceCounter(&nBeginTime); 
	//每次读取1个数据包，即17samples*16channels，数据类型为24bit整型，以int型存储
	for(int ii=0;ii<DataFileNum;ii++)
	{
		fileindex = ii;
		FileIdx = to_string(fileindex);
		FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
		if(fp != NULL)
		{
			fclose(fp);
			fp = NULL;
		}
		fp = fopen(FileName.c_str(),"rb");
		for(int jj=0;jj<8e4;jj++)
		{
			while(dftime < TL*1.0 / FS)
			{
				QueryPerformanceCounter(&nEndTime);
				dftime = (double)(nEndTime.QuadPart-nBeginTime.QuadPart)/(double)nFreq.QuadPart;
			}
			dftime = 0.0;
			nBeginTime = nEndTime;
			fread(readbuf,sizeof(int),TL*CHANNUM+1,fp);
			if(0 == BUF_FLAG)
			{
				pBuf = DataBufA_B1; 
				pCounter = &CounterA;
			}
			else
			{
				pBuf = DataBufB_B1; 
				pCounter = &CounterB;
			}
			if(*(pCounter)>=TL) // TL长的数据全部写入pBuf
			{
				memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,TL*CHANNUM*sizeof(int));
				*(pCounter) = *(pCounter)-TL;
			}
			else
			{
				temp = TL - *(pCounter);
				//写*(pCounter)个数据至pBuf
				memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,(*(pCounter))*CHANNUM*sizeof(int));
				//重置CounterA或CounterB计数器
				*(pCounter)= FRAMELEN;
				//写temp个数据至另一缓冲
				if(0 == BUF_FLAG) //当前为A，则写入B
				{
					memcpy(DataBufB_B1+FRAMELEN*CHANNUM-CounterB*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
					//修改B计数值
					CounterB = CounterB - temp;
					//切换缓冲
					BUF_FLAG = 1;
				}
				else //当前为B，则写入A
				{
					memcpy(DataBufA_B1+FRAMELEN*CHANNUM-CounterA*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
					//修改A计数值
					CounterA = CounterA - temp;
					//切换缓冲
					BUF_FLAG = 0;
				}
				//使事件有效
				SetEvent(g_hReadBoard1ThreadReadyEnvent);
			}
		}
	}
}

void ReadBoard2Data(LPVOID lParam)

{
	_Longlong fileindex = 0;
	string FilePath = "D:\\20180201宜昌鱼声阵数据\\20180201\\h\\uwrn\\";
	string FileNamePre = "Board2_ADC_";
	string FileIdx = to_string(fileindex);
	string FileNameSur = ".bin";
	string FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
	int DataFileNum = 18;
	FILE *fp = NULL;
	LARGE_INTEGER nFreq;
    LARGE_INTEGER nBeginTime;
    LARGE_INTEGER nEndTime;
	double dftime = 0.0;
	int readbytes = 0;
	int readbuf[TL*CHANNUM+1];
	int BUF_FLAG=0;
	int *pBuf = NULL;
	int *pCounter = NULL;
	int CounterA = FRAMELEN,CounterB = FRAMELEN;
	int temp = 0;

	QueryPerformanceFrequency(&nFreq);

	if(DataBufA_B2 != NULL)
	{
		free(DataBufA_B2);
		DataBufA_B2 = NULL;
	}
	DataBufA_B2 = (int *)malloc(FRAMELEN*CHANNUM*sizeof(int));
	memset(DataBufA_B2,0,FRAMELEN*CHANNUM*sizeof(int));

	if(DataBufB_B2 != NULL)
	{
		free(DataBufB_B2);
		DataBufB_B2 = NULL;
	}
	DataBufB_B2 = (int *)malloc(FRAMELEN*CHANNUM*sizeof(int));
	memset(DataBufB_B2,0,FRAMELEN*CHANNUM*sizeof(int));

	//每次读取1个数据包，即17samples*16channels，数据类型为24bit整型，以int型存储
	QueryPerformanceCounter(&nBeginTime); 
	for(int ii=0;ii<DataFileNum;ii++)
	{
		fileindex = ii;
		FileIdx = to_string(fileindex);
		FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
		if(fp != NULL)
		{
			fclose(fp);
			fp = NULL;
		}
		fp = fopen(FileName.c_str(),"rb");
		for(int jj=0;jj<8e4;jj++)
		{
			while(dftime < TL*1.0 / FS)
			{
				QueryPerformanceCounter(&nEndTime);
				dftime = (double)(nEndTime.QuadPart-nBeginTime.QuadPart)/(double)nFreq.QuadPart;
			}
			dftime = 0.0;
			nBeginTime = nEndTime;
			fread(readbuf,sizeof(int),TL*CHANNUM+1,fp);
			if(0 == BUF_FLAG)
			{
				pBuf = DataBufA_B2; 
				pCounter = &CounterA;
			}
			else
			{
				pBuf = DataBufB_B2; 
				pCounter = &CounterB;
			}
			if(*(pCounter)>=TL) // TL长的数据全部写入pBuf
			{
				memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,TL*CHANNUM*sizeof(int));
				*(pCounter) = *(pCounter)-TL;
			}
			else
			{
				temp = TL - *(pCounter);
				//写*(pCounter)个数据至pBuf
				memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,(*(pCounter))*CHANNUM*sizeof(int));
				//重置CounterA或CounterB计数器
				*(pCounter)= FRAMELEN;
				//写temp个数据至另一缓冲
				if(0 == BUF_FLAG) //当前为A，则写入B
				{
					memcpy(DataBufB_B2+FRAMELEN*CHANNUM-CounterB*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
					//修改B计数值
					CounterB = CounterB - temp;
					//切换缓冲
					BUF_FLAG = 1;
				}
				else //当前为B，则写入A
				{
					memcpy(DataBufA_B2+FRAMELEN*CHANNUM-CounterA*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
					//修改A计数值
					CounterA = CounterA - temp;
					//切换缓冲
					BUF_FLAG = 0;
				}
				//使事件有效
				SetEvent(g_hReadBoard2ThreadReadyEnvent);
			}
		}
	}	
}

void DataFormatting(LPVOID lParam)
{
	int retval1 = -1;
	int retval2 = -1;
	int BUF_FLAG = 0;
	int temp = 0;

	if(ChannDataBufA != NULL)
	{
		free(ChannDataBufA);
		ChannDataBufA = NULL;
	}
	ChannDataBufA = (float *)malloc(FRAMELEN*CHANNUM*2*sizeof(float));
	memset(ChannDataBufA,0,FRAMELEN*CHANNUM*2*sizeof(float));

	if(ChannDataBufB != NULL)
	{
		free(ChannDataBufB);
		ChannDataBufB = NULL;
	}
	ChannDataBufB = (float *)malloc(FRAMELEN*CHANNUM*2*sizeof(float));
	memset(ChannDataBufB,0,FRAMELEN*CHANNUM*2*sizeof(float));

	while(1)
	{
		retval1 = WaitForSingleObject(g_hReadBoard1ThreadReadyEnvent,2000);
		retval2 = WaitForSingleObject(g_hReadBoard2ThreadReadyEnvent,2000);
		if(retval1 == WAIT_OBJECT_0 && retval2 == WAIT_OBJECT_0)
		{
			if(BUF_FLAG == 0)
			{
				for(int ii=0;ii<CHANNUM;ii++)
				{
					for(int jj=0;jj<FRAMELEN;jj++)
					{
						temp = DataBufA_B1[jj*CHANNUM+ii];
						temp = temp<<8;
						temp = temp>>8;
						ChannDataBufA[ii*FRAMELEN+jj] = temp*1.0/pow(2.0,23) * 2.5;
						
						temp = DataBufA_B2[jj*CHANNUM+ii];
						temp = temp<<8;
						temp = temp>>8;
						ChannDataBufA[ii*FRAMELEN+jj+FRAMELEN*CHANNUM] = temp*1.0/pow(2.0,23) * 2.5;
					}
				}
				BUF_FLAG = 1;
				SetEvent(g_hFrameDataReadyEnvent);
			}
			else
			{
				for(int ii=0;ii<CHANNUM;ii++)
				{
					for(int jj=0;jj<FRAMELEN;jj++)
					{
						temp = DataBufB_B1[jj*CHANNUM+ii];
						temp = temp<<8;
						temp = temp>>8;
						ChannDataBufB[ii*FRAMELEN+jj] = temp*1.0/pow(2.0,23) * 2.5;
						
						temp = DataBufB_B2[jj*CHANNUM+ii];
						temp = temp<<8;
						temp = temp>>8;
						ChannDataBufB[ii*FRAMELEN+jj+FRAMELEN*CHANNUM] = temp*1.0/pow(2.0,23) * 2.5;
					}
				}
				BUF_FLAG = 0;
				SetEvent(g_hFrameDataReadyEnvent);
			}
		}
		else
		{
			printf("DataRead Timeout!\n");
		}
	}
}

//void ArraySignalProcessing(LPVOID lParam)
//{
//	int retval = -1;
//	int BUF_FLAG = 0;
//	int FrameNum = 0;
//	
//	//-----------------滤波降采样参数-------------------------------
//	float h[FIRORDER+1] = {0.0};
//	float fl = 100.0f,fh = 10e3f;	
//	cudaError    cudaStatus;
//	cufftReal    *dev_x=NULL;              //32通道原始数据
//	cufftReal    *dev_h=NULL;              //滤波器系数
//	cufftComplex *dev_fft_x=NULL;          //32通道原始数据FFT
//	cufftComplex *dev_fft_h=NULL;          //滤波器系数FFT
//	cufftComplex *dev_fft_y=NULL;          //滤波器输出FFT
//	cufftReal    *dev_y=NULL;              //滤波器输出原始采样率时域信号
//	cufftReal    *dev_chanbuff=NULL;       //显存内数据缓冲区
//	float        *FilteredDataout = NULL;
//	float        *DownSamplingData = NULL;
//	cufftHandle  Hplan;                    //滤波器系数FFT
//	cufftHandle  Xplan;                    //通道原始数据FFT
//	cufftHandle  Yplan;                    //滤波后通道数据FFT
//	//----------------------------------------------------------------
//
//	//--------------------------测时延变量----------------------------
//	LARGE_INTEGER nFreq;
//    LARGE_INTEGER nBeginTime;
//    LARGE_INTEGER nEndTime;
//	double time;
//	cudaEvent_t start1;
//	cudaEvent_t stop1;
//	float msecTotal = 0.0f;
//	//----------------------------------------------------------------
//
//	//--------------------------频域波束形成参数----------------------
//	int nfl = (int)((2000.0/(FS/DOWNSAMPLE)*NFFT)+0.5);
//	int nfh = (int)((4000.0/(FS/DOWNSAMPLE)*NFFT)+0.5);
//	int FreqbinPerThread = (int)((nfh-nfl+1)/(THREADNUMPERBLK*1.0) + 0.5);
//	int BlockRowNum = 0;
//	cufftComplex    *dev_fft=NULL;         //32通道降采样信号FFT
//	cufftReal       *dev_energy=NULL;      //分频段波束能量，每个频段512个频点
//	cufftReal       *sum_energy=NULL;      //全频段波束能量，频段外的能量置为零
//	cufftComplex    *PhiArray = NULL;      //各阵元各频点相移因子
//	cufftHandle     Beamplan;              //频域波束形成FFT
//	float           c[BEAMNUM]={0.0};      //调试用
//	Complex         *sk=NULL;
//	float           *debugvar = NULL;
//	int             peak[BEAMNUM]={0};
//	int             valley[BEAMNUM]={0};
//	bool            traced[BEAMNUM] = {false};
//	int             tracedbeamIdx = -1;
//	float           pretracedtarget[BEAMNUM] = {0.0};
//	int             pretracedtargetIdx[BEAMNUM] = {-1};
//	int             pretracedtargetNum = 0;
//	int             tracedtargetbeam[MAXTRACETARNUM][2];
//	float           *tracebeam = NULL;
//	int             beammatrix[5][BEAMNUM] = {0};
//	int             i0,i1,i2;
//	float           r0,r1,r2;
//	float           delta_index = 0;  
//	float           tracedtargetangle[3] = {0.0};
//	cufftReal       *dev_delayFilter = NULL;    //各通道时延滤波器系数
//	cufftReal       *dev_tau = NULL;
//	float           delayfiltercoff[ARRAYNUM*(2*M+1)] = {0.0};
//	float           delaytau[ARRAYNUM] = {0.0};
//	cufftReal       *dev_delayfilterout = NULL;
//	cufftReal       *dev_delayfilterbuf = NULL;
//	int             *dev_dI = NULL;
//	int             delaydI[ARRAYNUM] = {0};
//	float           *sourcedata = NULL;
//	float           *shiftdata = NULL;
//	float           *delayfilteroutdata = NULL;
//	cufftReal       *dev_delaychandata = NULL;
//	cufftReal       *dev_beamdata = NULL;
//	float           *beamdata = NULL;
//	//----------------------------------------------------------------
//
//	if(DownSamplingDataBufA != NULL)
//	{
//		free(DownSamplingDataBufA);
//		DownSamplingDataBufA = NULL;
//	}
//	DownSamplingDataBufA = (float *)malloc(FILTER_FRAME*CHANNUM*2*sizeof(float));
//	memset(DownSamplingDataBufA,0,FILTER_FRAME*CHANNUM*2*sizeof(float));
//
//	if(DownSamplingDataBufB != NULL)
//	{
//		free(DownSamplingDataBufB);
//		DownSamplingDataBufB = NULL;
//	}
//	DownSamplingDataBufB = (float *)malloc(FILTER_FRAME*CHANNUM*2*sizeof(float));
//	memset(DownSamplingDataBufB,0,FILTER_FRAME*CHANNUM*2*sizeof(float));
//
//
//	//-----------------调试用-----------------------------------
//	FilteredDataout = (float *)malloc(FILTER_FRAME/DOWNSAMPLE*sizeof(float));
//	memset(FilteredDataout,0,FILTER_FRAME/DOWNSAMPLE*sizeof(float));
//	DownSamplingData = (float *)malloc(FRAMELEN*sizeof(float));
//	memset(DownSamplingData,0,FRAMELEN*sizeof(float));
//
//	Complex *Xk_real = NULL;
//	Xk_real = (Complex *)malloc(FILTER_FRAME*sizeof(Complex));
//	memset(Xk_real,0,FILTER_FRAME*sizeof(Complex));
//
//	FILE *fp = NULL;
//	fp = fopen("BeamEng.bin","wb");
//	FILE *fplog = NULL;
//	fplog = fopen("ProcessLog.txt","w");
//	FILE *fpbeam = NULL;
//	fpbeam = fopen("Beam.bin","wb");
//	int retvalprint = 0;
//
//	//-----------------调试用-----------------------------------
//	
//    cufftPlan1d(&Hplan, FILTER_FRAME, CUFFT_R2C, 1);  
//    cufftPlan1d(&Xplan, FILTER_FRAME, CUFFT_R2C, 1);  
//    cufftPlan1d(&Yplan, FILTER_FRAME, CUFFT_C2R, 1);  
//
//	cudaStatus = cudaMalloc((void **)&dev_x, sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf (" dev_x cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_x,0,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);
//
//	cudaStatus = cudaMalloc((void **)&dev_h, sizeof(cufftReal)*FILTER_FRAME);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_h cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_h,0,sizeof(cufftReal)*FILTER_FRAME);
//
//	cudaStatus = cudaMalloc((void **)&dev_y, sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_y cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_y,0,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);
//
//	cudaStatus = cudaMalloc((void **)&dev_fft_x,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_fft_x cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_fft_x,0,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);
//
//	cudaStatus = cudaMalloc((void **)&dev_fft_h,sizeof(cufftComplex)*FILTER_FRAME);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_fft_h cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_fft_h,0,sizeof(cufftComplex)*FILTER_FRAME);
//
//	cudaStatus = cudaMalloc((void **)&dev_fft_y,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_fft_y cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_fft_y,0,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);
//
//	cudaStatus = cudaMalloc((void **)&dev_chanbuff,sizeof(cufftReal)*FILTER_FRAME/DOWNSAMPLE*CHANNUM*2);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_chanbuff cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_chanbuff,0,sizeof(cufftReal)*FILTER_FRAME/DOWNSAMPLE*CHANNUM*2);
//
//	fir1(FIRORDER,3,fl,fh,FS,5,h);
//	cudaMemcpy(dev_h,h,sizeof(cufftReal)*FIRORDER,cudaMemcpyHostToDevice);
//	cufftExecR2C(Hplan,(cufftReal *)&dev_h[0],(cufftComplex *)&dev_fft_h[0]);
//
//	BlockRowNum = NFFT/2/THREADNUMPERBLK;
//	cudaStatus = cudaMalloc((void**)&dev_energy,BEAMNUM*BlockRowNum*sizeof(cufftReal));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_energy cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_energy,0,BEAMNUM*BlockRowNum*sizeof(cufftReal));
//
//	cudaStatus = cudaMalloc((void**)&sum_energy,BEAMNUM*sizeof(cufftReal));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("sum_energy cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&sum_energy,0,BEAMNUM*sizeof(cufftReal));
//
//	cudaStatus = cudaMalloc((void**)&PhiArray,ARRAYNUM*BEAMNUM*(NFFT/2)*sizeof(cufftComplex));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("PhiArray cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&PhiArray,0,ARRAYNUM*BEAMNUM*(NFFT/2)*sizeof(cufftComplex));
//
//	cudaStatus = cudaMalloc((void **)&dev_fft,sizeof(cufftComplex)*(NFFT/2+1)*ARRAYNUM);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_fft cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_fft,0,sizeof(cufftComplex)*(NFFT/2+1)*ARRAYNUM);
//
//	cufftPlan1d(&Beamplan,NFFT,CUFFT_R2C, 1);
//
//	PhiShiftFactorGen<<<NFFT/2,BEAMNUM>>>(PhiArray);
//
//
//	sk = (Complex *)malloc(sizeof(Complex)*(NFFT/2+1)*ARRAYNUM);
//	memset(sk,0,sizeof(Complex)*(NFFT/2+1)*ARRAYNUM);
//
//	debugvar = (float *)malloc(sizeof(float)*BEAMNUM*BlockRowNum);
//	memset(debugvar,0, sizeof(float)*BEAMNUM*BlockRowNum);
//
//	for(int ii = 0;ii<MAXTRACETARNUM;ii++)
//	{
//		tracedtargetbeam[ii][0] = -1;
//		tracedtargetbeam[ii][1] = -1;
//		tracedtargetangle[ii] = -1.0f;
//	}
//
//	cudaStatus = cudaMalloc((void **)&dev_delayFilter,sizeof(cufftReal)*(2*M+1)*ARRAYNUM);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_delayFilter cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_delayFilter,0,sizeof(cufftReal)*(2*M+1)*ARRAYNUM);
//
//	cudaStatus = cudaMalloc((void **)&dev_tau,sizeof(cufftReal)*ARRAYNUM);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_tau cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_tau,0,sizeof(cufftReal)*ARRAYNUM);
//
//	cudaStatus = cudaMalloc((void **)&dev_delayfilterout,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE+2*M));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_delayfilterout cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_delayfilterout,0,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE+2*M));
//
//	cudaStatus = cudaMalloc((void **)&dev_delayfilterbuf,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_delayfilterbuf cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_delayfilterbuf,0,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE));
//
//	cudaStatus = cudaMalloc((void **)&dev_dI,sizeof(int)*ARRAYNUM);
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_dI cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_dI,0,sizeof(int)*ARRAYNUM);
//
//	cudaStatus = cudaMalloc((void **)&dev_delaychandata,sizeof(int)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE/2));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_delaychandata cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_delaychandata,0,sizeof(int)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE/2));
//
//	cudaStatus = cudaMalloc((void **)&dev_beamdata,sizeof(int)*MAXTRACETARNUM*(FILTER_FRAME/DOWNSAMPLE/2));
//	if (cudaStatus != cudaSuccess)
//	{
//		printf ("dev_beamdata cudaMalloc Error! \n ");
//	}
//	cudaMemset((void **)&dev_beamdata,0,sizeof(int)*MAXTRACETARNUM*(FILTER_FRAME/DOWNSAMPLE/2));
//	
//
//	sourcedata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE)*sizeof(float));
//	memset(sourcedata,0,(FILTER_FRAME/DOWNSAMPLE)*sizeof(float));
//
//	shiftdata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE)*sizeof(float));
//	memset(shiftdata,0,(FILTER_FRAME/DOWNSAMPLE)*sizeof(float));
//
//	delayfilteroutdata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE+2*M)*sizeof(float));
//	memset(delayfilteroutdata,0,(FILTER_FRAME/DOWNSAMPLE+2*M)*sizeof(float));	
//
//	beamdata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float));
//	memset(beamdata,0,(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float));
//
//	QueryPerformanceFrequency(&nFreq);
//	cudaEventCreate(&start1);
//	cudaEventCreate(&stop1);
//
//	while(1)
//	{
//		retval = WaitForSingleObject(g_hFrameDataReadyEnvent,2000);
//		FrameNum++;
//		
//		if(retval<0)
//		{
//			printf("Timeout!\n");
//			return;
//		}
//
//		//移动缓冲区
//		if(BUF_FLAG == 0)
//		{
//			for(int ii=0;ii<CHANNUM*2;ii++)
//			{
//				memmove(DownSamplingDataBufA+ii*FILTER_FRAME,DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,FRAMELEN*sizeof(float));
//				memcpy(DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,ChannDataBufA+ii*FRAMELEN,FRAMELEN*sizeof(float));
//			}
//			cudaMemcpy(dev_x,DownSamplingDataBufA,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2,cudaMemcpyHostToDevice);
//			BUF_FLAG = 1;
//		}
//		else
//		{
//			for(int ii=0;ii<CHANNUM*2;ii++)
//			{
//				memmove(DownSamplingDataBufA+ii*FILTER_FRAME,DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,FRAMELEN*sizeof(float));
//				memcpy(DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,ChannDataBufB+ii*FRAMELEN,FRAMELEN*sizeof(float));
//			}
//			cudaMemcpy(dev_x,DownSamplingDataBufA,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2,cudaMemcpyHostToDevice);
//			BUF_FLAG = 0;
//		}
//		
//		cudaEventRecord(start1,NULL);
//
//		//-----------------------------------------(1) 信号滤波降采样---------------------------------------------------
//		//4.7ms
//		for(int jj=0;jj<CHANNUM*2;jj++)
//		{
//			cufftExecR2C(Xplan,(cufftReal *)&dev_x[jj*FILTER_FRAME],(cufftComplex *)&dev_fft_x[jj*FILTER_FRAME]);
//		}
//		
//		//频域相乘(13ms)
//		DownSamplingFilter<<<CHANNUM*2*(FILTER_FRAME/2/THREADNUMPERBLK),THREADNUMPERBLK>>>(dev_fft_x,dev_fft_h,dev_fft_y,FILTER_FRAME);
//	
//		QueryPerformanceCounter(&nBeginTime); 
//		//反变换(105ms)
//		for(int jj=0;jj<CHANNUM*2;jj++)
//		{
//			cufftExecC2R(Yplan,(cufftComplex *)&dev_fft_y[jj*FILTER_FRAME],(cufftReal *)&dev_y[jj*FILTER_FRAME]);
//			cudaMemcpy(dev_chanbuff+jj*FILTER_FRAME/DOWNSAMPLE,dev_chanbuff+jj*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2,FILTER_FRAME/DOWNSAMPLE/2*sizeof(float),cudaMemcpyDeviceToDevice);
//		}
//		IFFTNormalize<<<CHANNUM*2*(FILTER_FRAME/2/THREADNUMPERBLK),THREADNUMPERBLK>>>(dev_y,dev_chanbuff,FILTER_FRAME);	
//
//		QueryPerformanceCounter(&nEndTime);
//		//-----------------------------------------(1) 信号滤波降采样结束---------------------------------------------------
//
//
//		//-----------------------------------------(2) 频域波束形成---------------------------------------------------
//
//		//使用缓冲区中的后FILTER_FRAME/DOWNSAMPLE/2点数据做频域波束形成，估计方位
//		for (int ii=0;ii<ARRAYNUM;ii++)		
//		{		
//			cufftExecR2C(Beamplan,(cufftReal *)&dev_chanbuff[ii*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2],(cufftComplex *)&dev_fft[ii*(NFFT/2+1)]);
//		}
//
//		FD_Beamform<<<BlockRowNum*BEAMNUM,THREADNUMPERBLK>>>(dev_fft,dev_energy,PhiArray,nfl,nfh);//波束形成
//		MatrixSumRow<<<BEAMNUM,1>>>(dev_energy,sum_energy,BlockRowNum,BEAMNUM);
//		
//		cudaMemcpy(c,sum_energy,BEAMNUM*sizeof(float),cudaMemcpyDeviceToHost);
//		fwrite(c,sizeof(float),BEAMNUM,fp);
//		//-----------------------------------------(2) 频域波束形成结束-----------------------------------------------
//
//
//		//-----------------------------------------(3) 波束能量检测------------------------------------------
//		//波束能量检测与跟踪
//		memset(peak,0,BEAMNUM*sizeof(int));
//		memset(valley,0,BEAMNUM*sizeof(int));
//		findpeak(c,peak,BEAMNUM);
//		findvalley(c,valley,BEAMNUM);
//		bool targetexist = false;
//		//memmove(beammatrix+BEAMNUM,beammatrix,4*BEAMNUM*sizeof(int));
//		memset(pretracedtarget,0,sizeof(float)*BEAMNUM);
//		memset(pretracedtargetIdx,0,sizeof(int)*BEAMNUM);
//		pretracedtargetNum = 0;
//
//		for(int kk=0;kk<BEAMNUM;kk++)
//		{
//			if(peak[kk] == 1)
//			{
//				//判断是否已跟踪该波束附近目标
//				int jj=0;
//				for(jj=0;jj<MAXTRACETARNUM;jj++)
//				{
//					//先找是否已跟踪
//					if(abs(tracedtargetbeam[jj][0]-kk)<6 && tracedtargetbeam[jj][0]>0)   //已跟踪该目标，更新跟踪器角度
//					{
//						break;
//					}
//				}
//				if(jj==MAXTRACETARNUM)  //未跟踪
//				{
//					targetexist = peakdetection(kk,c,valley,2.0);
//				}
//				else  //已跟踪，降低检测门限
//				{
//					targetexist = peakdetection(kk,c,valley,1.2);
//				}
//				if(targetexist)
//				{
//					pretracedtarget[pretracedtargetNum] = c[kk];
//					pretracedtargetIdx[pretracedtargetNum] = kk;
//					pretracedtargetNum++;
//				}
//			}
//		}
//		rbub(pretracedtarget,pretracedtargetIdx,BEAMNUM);
//
//		if(FrameNum == 115)
//		{
//			FrameNum = FrameNum;
//		}
//		for(int kk=0;kk<pretracedtargetNum;kk++)
//		{
//			int jj=0;
//			for(jj=0;jj<MAXTRACETARNUM;jj++)
//			{
//				//先找是否已跟踪
//				if(abs(tracedtargetbeam[jj][0]-pretracedtargetIdx[kk])<6 && tracedtargetbeam[jj][0]>0)   //已跟踪该目标，更新跟踪器角度
//				{
//					tracedtargetbeam[jj][0] = pretracedtargetIdx[kk];
//					tracedtargetbeam[jj][1] = FrameNum;
//					break;
//				}
//			}
//
//			if(jj==MAXTRACETARNUM)  //未跟踪该目标，找一个空的跟踪器跟踪
//			{
//				int ii = 0;
//				for(ii=0;ii<MAXTRACETARNUM;ii++)
//				{
//					//先找是否已跟踪
//					if(tracedtargetbeam[ii][0] < 0)
//					{
//						break;
//					}
//				}
//				if(ii < MAXTRACETARNUM)           //有空置跟踪器
//				{
//					tracedtargetbeam[ii][0] = pretracedtargetIdx[kk];
//					tracedtargetbeam[ii][1] = FrameNum;
//				}
//			}
//		}
//		//跟踪器管理，清空多帧未更新的跟踪器
//		for(int jj=0;jj<MAXTRACETARNUM;jj++)
//		{
//			if(tracedtargetbeam[jj][0] >0 && FrameNum - tracedtargetbeam[jj][1] >= 5)
//			{
//				tracedtargetbeam[jj][0] = -1;
//				tracedtargetbeam[jj][1] = -1;
//				tracedtargetangle[jj] = -1.0f;
//			}
//		}
//		//-----------------------------------------(3) 波束能量检测-------------------------------------
//
//
//		//-----------------------------------------(4) 波束跟踪、跟踪波束 ------------------------------
//		for(int jj = 0;jj<MAXTRACETARNUM;jj++)
//		{
//			if(tracedtargetbeam[jj][0] >0)   //有跟踪目标
//			{
//				//波束内插
//				i0 = tracedtargetbeam[jj][0]-1;
//				i1 = tracedtargetbeam[jj][0];
//				i2 = tracedtargetbeam[jj][0]+1;
//				r0 = c[i0];
//				r1 = c[i1];
//				r2 = c[i2];
//				delta_index = (r2-r0)/(4*r1-2*r0-2*r2);
//				tracedtargetangle[jj] = (i1+delta_index)*180.0/BEAMNUM;
//				DelayFilterGen<<<ARRAYNUM,2*M+1>>>(dev_delayFilter,M,tracedtargetangle[jj],dev_tau,dev_dI);
//				//DelayFilterGen<<<ARRAYNUM,2*M+1>>>(dev_delayFilter,M,60.292690,dev_tau,dev_dI);
//				cudaMemcpy(delayfiltercoff,dev_delayFilter,sizeof(cufftReal)*ARRAYNUM*(2*M+1),cudaMemcpyDeviceToHost);
//				cudaMemcpy(delaytau,dev_tau,sizeof(cufftReal)*ARRAYNUM,cudaMemcpyDeviceToHost);
//				cudaMemcpy(delaydI,dev_dI,sizeof(int)*ARRAYNUM,cudaMemcpyDeviceToHost);
//				
//				for(int kk = 0;kk<ARRAYNUM;kk++)
//				{
//					if(delaydI[kk] >= 0)
//					{
//						cudaMemcpy(dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE)+delaydI[kk],dev_chanbuff+kk*(FILTER_FRAME/DOWNSAMPLE),sizeof(cufftReal)*((FILTER_FRAME/DOWNSAMPLE)-delaydI[kk]),cudaMemcpyDeviceToDevice);
//					}
//					else
//					{
//						cudaMemcpy(dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE),dev_chanbuff+kk*(FILTER_FRAME/DOWNSAMPLE)-delaydI[kk],sizeof(cufftReal)*((FILTER_FRAME/DOWNSAMPLE)+delaydI[kk]),cudaMemcpyDeviceToDevice);
//					}
//
//					//cudaMemcpy(sourcedata,dev_chanbuff+kk*(FILTER_FRAME/DOWNSAMPLE),(FILTER_FRAME/DOWNSAMPLE)*sizeof(float),cudaMemcpyDeviceToHost);
//					//cudaMemcpy(shiftdata,dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE),(FILTER_FRAME/DOWNSAMPLE)*sizeof(float),cudaMemcpyDeviceToHost);
//
//					if(fabs(delaytau[kk]) > 0.0001)
//					{
//						FineDelayFilter<<<(FILTER_FRAME/DOWNSAMPLE+2*M),2*M+1>>>((cufftReal *)&dev_delayfilterbuf[kk*FILTER_FRAME/DOWNSAMPLE],(cufftReal *)&dev_delayfilterout[kk*(FILTER_FRAME/DOWNSAMPLE+2*M)],(cufftReal *)&dev_delayFilter[kk*(2*M+1)],M);
//					}
//					else
//					{
//						cudaMemcpy(dev_delayfilterout+kk*(FILTER_FRAME/DOWNSAMPLE+2*M)+M,dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE),sizeof(cufftReal)*(FILTER_FRAME/DOWNSAMPLE),cudaMemcpyDeviceToDevice);
//					}
//					cudaMemcpy(dev_delaychandata+kk*(FILTER_FRAME/DOWNSAMPLE/2),dev_delayfilterout+kk*(FILTER_FRAME/DOWNSAMPLE+2*M)+M+FILTER_FRAME/DOWNSAMPLE/4,sizeof(cufftReal)*FILTER_FRAME/DOWNSAMPLE/2,cudaMemcpyDeviceToDevice);
//					//cudaMemcpy(delayfilteroutdata,dev_delayfilterout+kk*(FILTER_FRAME/DOWNSAMPLE+2*M),(FILTER_FRAME/DOWNSAMPLE+M*2)*sizeof(float),cudaMemcpyDeviceToHost);					
//					//if(FrameNum==2)
//					//{
//					//	FrameNum = FrameNum;
//					//}
//				}
//				MatrixSumRow<<<FILTER_FRAME/DOWNSAMPLE/2,1>>>(dev_delaychandata,dev_beamdata+jj*FILTER_FRAME/DOWNSAMPLE/2,ARRAYNUM,FILTER_FRAME/DOWNSAMPLE/2);
//				cudaMemcpy(beamdata,dev_beamdata+jj*FILTER_FRAME/DOWNSAMPLE/2,FILTER_FRAME/DOWNSAMPLE/2*sizeof(float),cudaMemcpyDeviceToHost);
//				fwrite(beamdata,sizeof(float),FILTER_FRAME/DOWNSAMPLE/2,fpbeam);
//			}
//		}
//
//		cudaEventRecord(stop1,NULL);
//		cudaEventSynchronize(stop1);
//		//time=(double)(nEndTime.QuadPart-nBeginTime.QuadPart)/(double)nFreq.QuadPart;
//		cudaEventElapsedTime(&msecTotal,start1,stop1);
//		printf("%d:%f;%d,%d;%d,%d;%d,%d\n",FrameNum,msecTotal,tracedtargetbeam[0][0],tracedtargetbeam[0][1],tracedtargetbeam[1][0],tracedtargetbeam[1][1],tracedtargetbeam[2][0],tracedtargetbeam[2][1]);
//		fprintf(fplog,"%d:%f;%d,%d;%d,%d;%d,%d\n",FrameNum,msecTotal,tracedtargetbeam[0][0],tracedtargetbeam[0][1],tracedtargetbeam[1][0],tracedtargetbeam[1][1],tracedtargetbeam[2][0],tracedtargetbeam[2][1]);
//		fflush(fplog);
//	}
//	fclose(fp);
//	fp = NULL;
//	fclose(fplog);
//	fplog = NULL;
//	fclose(fpbeam);
//	fpbeam = NULL;
//}
__global__ void Psd(cufftComplex *Xk,cufftReal *Xabs, int N)
{
    int bid = 0,tid = 0;
    int freqIdx = 0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    freqIdx = bid*THREADNUMPERBLK+tid;

    Xabs[freqIdx] = (Xk[freqIdx].x*Xk[freqIdx].x+Xk[freqIdx].y*Xk[freqIdx].y) / N;
}

__global__ void PsdAverage(cufftReal *Xabs,cufftReal *Xk_avg)
{
    int bid = 0,tid = 0;
    int freqIdx = 0;
	float sum = 0.0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    freqIdx = bid*THREADNUMPERBLK+tid;

	for(int ii = 0;ii<PSD_AVG_NUM;ii++)
	{
		sum += Xabs[ii*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)+freqIdx] / PSD_AVG_NUM;
	}
	Xk_avg[freqIdx] = 10*log10((sum+EPS)/1e-12);
}

__global__ void PsdSub(cufftReal *Xk_avg,cufftReal *Xk_smooth,cufftReal *Xk_diff,int idx1,int idx2)
{
    int bid = 0,tid = 0;
    int freqIdx = 0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    freqIdx = bid*THREADNUMPERBLK+tid;
	if(freqIdx >= idx1 && freqIdx <= idx2)
	{
		Xk_diff[freqIdx] = Xk_avg[freqIdx] - Xk_smooth[freqIdx];
	}
	else
	{
		Xk_diff[freqIdx] = 0;
	}
	//if(freqIdx == 50000)
	//{
	//	printf("Xk_smooth=%.5f\n",Xk_smooth[freqIdx]);
	//}
}
//__global__ void PsdLog(cufftReal *Xk_avg)
//{
//    int bid = 0,tid = 0;
//    int freqIdx = 0;
//	float sum = 0.0;
//
//    bid = blockIdx.x;
//    tid = threadIdx.x;
//
//    freqIdx = bid*THREADNUMPERBLK+tid;
//
//	Xk_avg[freqIdx] = 10*log10(Xk_avg[freqIdx]+EPS);
//}

__global__ void FrequencyDomainFilter(cufftComplex *Xk,float deltaf,float StartFreq,float EndFreq)
{
    int bid = 0,tid = 0;
    int freqIdx = 0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    freqIdx = bid*THREADNUMPERBLK+tid;
	if(freqIdx * deltaf < StartFreq || freqIdx * deltaf > EndFreq)
	{
		Xk[freqIdx].x = 0;
		Xk[freqIdx].y = 0;
	}
	//else
	//{
	//	printf("Xk[freqIdx].x = %.6f\n",Xk[freqIdx].x);
	//}
}

__global__ void SignalSqr(cufftReal *X)
{
    int bid = 0,tid = 0;
    int sigIdx = 0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    sigIdx = bid*THREADNUMPERBLK+tid;
	X[sigIdx] = X[sigIdx]*X[sigIdx];
}

__global__ void DemonAdd(cufftComplex *Xk,cufftReal *Xabs, int N)
{
    int bid = 0,tid = 0;
    int freqIdx = 0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    freqIdx = bid*THREADNUMPERBLK+tid;

    Xabs[freqIdx] += (Xk[freqIdx].x*Xk[freqIdx].x+Xk[freqIdx].y*Xk[freqIdx].y) / N;
}

__global__ void DemonSub(cufftReal *Xk_avg,cufftReal *Xk_smooth,cufftReal *Xk_diff)
{
    int bid = 0,tid = 0;
    int freqIdx = 0;

    bid = blockIdx.x;
    tid = threadIdx.x;

    freqIdx = bid;
	Xk_diff[freqIdx] = Xk_avg[freqIdx] - Xk_smooth[freqIdx];
	if(Xk_diff[freqIdx] < 0)
	{
		Xk_diff[freqIdx] = 0;
	}
}

float VectorThetSPF(cufftComplex P_f, cufftComplex Vx_f, cufftComplex Vy_f)
{
	float fTheta=0.0;
	float sina=-P_f.y*Vy_f.x+P_f.x*Vy_f.y;
	float cosa=-P_f.y*Vx_f.x+P_f.x*Vx_f.y;
	fTheta=atan2(sina, cosa)*180/PI;
	return fTheta;
}

void ArraySignalProcessing(LPVOID lParam)
{
	int retval = -1;
	int BUF_FLAG = 0;
	int FrameNum = 0;
	
	//-----------------滤波降采样参数-------------------------------
	float h[FIRORDER+1] = {0.0};
	float fl = 100.0f,fh = 10e3f;	
	cudaError    cudaStatus;
	cufftReal    *dev_x=NULL;              //32通道原始数据
	cufftReal    *dev_h=NULL;              //滤波器系数
	cufftComplex *dev_fft_x=NULL;          //32通道原始数据FFT
	cufftComplex *dev_fft_h=NULL;          //滤波器系数FFT
	cufftComplex *dev_fft_y=NULL;          //滤波器输出FFT
	cufftReal    *dev_y=NULL;              //滤波器输出原始采样率时域信号
	cufftReal    *dev_chanbuff=NULL;       //显存内数据缓冲区
	float        *FilteredDataout = NULL;
	float        *DownSamplingData = NULL;
	cufftHandle  Hplan;                    //滤波器系数FFT
	cufftHandle  Xplan;                    //通道原始数据FFT
	cufftHandle  Yplan;                    //滤波后通道数据FFT
	//----------------------------------------------------------------

	//--------------------------测时延变量----------------------------
	LARGE_INTEGER nFreq;
    LARGE_INTEGER nBeginTime;
    LARGE_INTEGER nEndTime;
	double time;
	cudaEvent_t start1;
	cudaEvent_t stop1;
	float msecTotal = 0.0f;
	//----------------------------------------------------------------

	//--------------------------频域波束形成参数----------------------
	int nfl = (int)((2000.0/(FS/DOWNSAMPLE)*NFFT)+0.5);
	int nfh = (int)((4000.0/(FS/DOWNSAMPLE)*NFFT)+0.5);
	int FreqbinPerThread = (int)((nfh-nfl+1)/(THREADNUMPERBLK*1.0) + 0.5);
	int BlockRowNum = 0;
	cufftComplex    *dev_fft=NULL;         //32通道降采样信号FFT
	cufftReal       *dev_energy=NULL;      //分频段波束能量，每个频段512个频点
	cufftReal       *sum_energy=NULL;      //全频段波束能量，频段外的能量置为零
	cufftComplex    *PhiArray = NULL;      //各阵元各频点相移因子
	cufftHandle     Beamplan;              //频域波束形成FFT
	float           c[BEAMNUM]={0.0};      //调试用
	Complex         *sk=NULL;
	float           *debugvar = NULL;
	int             peak[BEAMNUM]={0};
	int             valley[BEAMNUM]={0};
	bool            traced[BEAMNUM] = {false};
	int             tracedbeamIdx = -1;
	float           pretracedtarget[BEAMNUM] = {0.0};
	int             pretracedtargetIdx[BEAMNUM] = {-1};
	int             pretracedtargetNum = 0;
	int             tracedtargetbeam[MAXTRACETARNUM][2];
	float           *tracebeam = NULL;
	int             beammatrix[5][BEAMNUM] = {0};
	int             i0,i1,i2;
	float           r0,r1,r2;
	float           delta_index = 0;  
	float           tracedtargetangle[3] = {0.0};
	cufftReal       *dev_delayFilter = NULL;    //各通道时延滤波器系数
	cufftReal       *dev_tau = NULL;
	float           delayfiltercoff[ARRAYNUM*(2*M+1)] = {0.0};
	float           delaytau[ARRAYNUM] = {0.0};
	cufftReal       *dev_delayfilterout = NULL;
	cufftReal       *dev_delayfilterbuf = NULL;
	int             *dev_dI = NULL;
	int             delaydI[ARRAYNUM] = {0};
	float           *sourcedata = NULL;
	float           *shiftdata = NULL;
	float           *delayfilteroutdata = NULL;
	cufftReal       *dev_delaychandata = NULL;
	cufftReal       *dev_beamdata = NULL;
	float           *beamdata = NULL;
	//----------------------------------------------------------------
    //----------------------------Psd and DEMON-----------------------
    cufftReal       *dev_tracebeam=NULL;
    cufftComplex    *dev_tracebeam_spec=NULL;
    cufftReal       *dev_tracebeam_psd=NULL;
	cufftReal       *dev_tracebeam_psd_avg = NULL;
    cufftComplex    *dev_tracebeam_demonspec=NULL;
	cufftComplex    *dev_tracebeam_demonspec_band=NULL;
    cufftReal       *dev_tracebeam_demon=NULL;
    cufftReal       *dev_tracebeam_demon_band_data=NULL;
    cufftHandle     PSDplan;
    cufftHandle     DEMONplan;
	cufftHandle     DEMONBandplan;
    float           *trace_beam_psd = NULL;
	float           fDf;
	int             idx1;
	int             idx2;
	int             idxLen;
    float           *trace_beam_psd_smooth = NULL;
	cufftReal       *dev_tracebeam_psd_S = NULL;
	cufftReal       *dev_tracebeam_psd_E = NULL;
	float           fPsdEVar=0.0;
    float           *trace_beam_demon = NULL;
	float           *trace_beam_demon_smooth = NULL;
	cufftReal       *dev_trace_beam_demon_cut = NULL;
	cufftReal       *dev_tracebeam_demon_S = NULL;
	cufftReal       *dev_tracebeam_demon_E = NULL;
	float           fDemonEVar=0.0;
	//-----------------------矢量通道处理-----------------------------
    cufftReal       *dev_vector_p_buf=NULL;
    cufftReal       *dev_vector_x_buf=NULL;
    cufftReal       *dev_vector_y_buf=NULL;
    cufftComplex    *dev_vector_p_spec=NULL;
    cufftComplex    *dev_vector_x_spec=NULL;
    cufftComplex    *dev_vector_y_spec=NULL;
    cufftReal       *dev_vector_p_psd =NULL;
	cufftReal       *dev_vector_psd_avg=NULL;
    float           *vector_p_psd = NULL;
    float           *vector_p_psd_smooth = NULL;
	cufftReal       *dev_vector_p_psd_S = NULL;
	cufftReal       *dev_vector_p_psd_E = NULL;
	float           fVectorPsdEVar=0.0;
    //----------------------------------------------------------------
	if(DownSamplingDataBufA != NULL)
	{
		free(DownSamplingDataBufA);
		DownSamplingDataBufA = NULL;
	}
	DownSamplingDataBufA = (float *)malloc(FILTER_FRAME*CHANNUM*2*sizeof(float));
	memset(DownSamplingDataBufA,0,FILTER_FRAME*CHANNUM*2*sizeof(float));

	if(DownSamplingDataBufB != NULL)
	{
		free(DownSamplingDataBufB);
		DownSamplingDataBufB = NULL;
	}
	DownSamplingDataBufB = (float *)malloc(FILTER_FRAME*CHANNUM*2*sizeof(float));
	memset(DownSamplingDataBufB,0,FILTER_FRAME*CHANNUM*2*sizeof(float));


	//-----------------调试用-----------------------------------
	FilteredDataout = (float *)malloc(FILTER_FRAME/DOWNSAMPLE*sizeof(float));
	memset(FilteredDataout,0,FILTER_FRAME/DOWNSAMPLE*sizeof(float));
	DownSamplingData = (float *)malloc(FRAMELEN*sizeof(float));
	memset(DownSamplingData,0,FRAMELEN*sizeof(float));

	cufftComplex *Xk_real = NULL;
	Xk_real = (cufftComplex *)malloc(FILTER_FRAME*sizeof(cufftComplex));
	memset(Xk_real,0,FILTER_FRAME*sizeof(cufftComplex));

	FILE *fp = NULL;
	fp = fopen("BeamEng.bin","wb");
	FILE *fplog = NULL;
	fplog = fopen("ProcessLog.txt","w");
	FILE *fpbeam = NULL;
	fpbeam = fopen("Beam.bin","wb");
	int retvalprint = 0;

	//-----------------调试用-----------------------------------
	
    cufftPlan1d(&Hplan, FILTER_FRAME, CUFFT_R2C, 1);  
    cufftPlan1d(&Xplan, FILTER_FRAME, CUFFT_R2C, 1);  
    cufftPlan1d(&Yplan, FILTER_FRAME, CUFFT_C2R, 1);  
    cufftPlan1d(&PSDplan,   PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2), CUFFT_R2C, 1);
    cufftPlan1d(&DEMONplan, PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2), CUFFT_R2C, 1);
    cufftPlan1d(&DEMONBandplan, PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2), CUFFT_C2R, 1);  

	cudaStatus = cudaMalloc((void **)&dev_x, sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);
	if (cudaStatus != cudaSuccess)
	{
		printf (" dev_x cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_x,0,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);

	cudaStatus = cudaMalloc((void **)&dev_h, sizeof(cufftReal)*FILTER_FRAME);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_h cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_h,0,sizeof(cufftReal)*FILTER_FRAME);

	cudaStatus = cudaMalloc((void **)&dev_y, sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_y cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_y,0,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2);

	cudaStatus = cudaMalloc((void **)&dev_fft_x,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_fft_x cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_fft_x,0,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);

	cudaStatus = cudaMalloc((void **)&dev_fft_h,sizeof(cufftComplex)*FILTER_FRAME);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_fft_h cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_fft_h,0,sizeof(cufftComplex)*FILTER_FRAME);

	cudaStatus = cudaMalloc((void **)&dev_fft_y,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_fft_y cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_fft_y,0,sizeof(cufftComplex)*FILTER_FRAME*CHANNUM*2);

	cudaStatus = cudaMalloc((void **)&dev_chanbuff,sizeof(cufftReal)*FILTER_FRAME/DOWNSAMPLE*CHANNUM*2);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_chanbuff cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_chanbuff,0,sizeof(cufftReal)*FILTER_FRAME/DOWNSAMPLE*CHANNUM*2);

	fir1(FIRORDER,3,fl,fh,FS,5,h);
	cudaMemcpy(dev_h,h,sizeof(cufftReal)*FIRORDER,cudaMemcpyHostToDevice);
	cufftExecR2C(Hplan,(cufftReal *)&dev_h[0],(cufftComplex *)&dev_fft_h[0]);

	BlockRowNum = NFFT/2/THREADNUMPERBLK;
	cudaStatus = cudaMalloc((void**)&dev_energy,BEAMNUM*BlockRowNum*sizeof(cufftReal));
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_energy cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_energy,0,BEAMNUM*BlockRowNum*sizeof(cufftReal));

	cudaStatus = cudaMalloc((void**)&sum_energy,BEAMNUM*sizeof(cufftReal));
	if (cudaStatus != cudaSuccess)
	{
		printf ("sum_energy cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&sum_energy,0,BEAMNUM*sizeof(cufftReal));

	cudaStatus = cudaMalloc((void**)&PhiArray,ARRAYNUM*BEAMNUM*(NFFT/2)*sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		printf ("PhiArray cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&PhiArray,0,ARRAYNUM*BEAMNUM*(NFFT/2)*sizeof(cufftComplex));

	cudaStatus = cudaMalloc((void **)&dev_fft,sizeof(cufftComplex)*(NFFT/2+1)*ARRAYNUM);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_fft cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_fft,0,sizeof(cufftComplex)*(NFFT/2+1)*ARRAYNUM);

	cufftPlan1d(&Beamplan,NFFT,CUFFT_R2C, 1);

	PhiShiftFactorGen<<<NFFT/2,BEAMNUM>>>(PhiArray);


    sk = (cufftComplex *)malloc(sizeof(cufftComplex)*(NFFT/2+1)*ARRAYNUM);
    memset(sk,0,sizeof(cufftComplex)*(NFFT/2+1)*ARRAYNUM);

	debugvar = (float *)malloc(sizeof(float)*BEAMNUM*BlockRowNum);
	memset(debugvar,0, sizeof(float)*BEAMNUM*BlockRowNum);

	for(int ii = 0;ii<MAXTRACETARNUM;ii++)
	{
		tracedtargetbeam[ii][0] = -1;
		tracedtargetbeam[ii][1] = -1;
		tracedtargetangle[ii] = -1.0f;
	}

	cudaStatus = cudaMalloc((void **)&dev_delayFilter,sizeof(cufftReal)*(2*M+1)*ARRAYNUM);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_delayFilter cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_delayFilter,0,sizeof(cufftReal)*(2*M+1)*ARRAYNUM);

	cudaStatus = cudaMalloc((void **)&dev_tau,sizeof(cufftReal)*ARRAYNUM);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_tau cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_tau,0,sizeof(cufftReal)*ARRAYNUM);

	cudaStatus = cudaMalloc((void **)&dev_delayfilterout,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE+2*M));
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_delayfilterout cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_delayfilterout,0,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE+2*M));

	cudaStatus = cudaMalloc((void **)&dev_delayfilterbuf,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE));
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_delayfilterbuf cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_delayfilterbuf,0,sizeof(cufftReal)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE));

	cudaStatus = cudaMalloc((void **)&dev_dI,sizeof(int)*ARRAYNUM);
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_dI cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_dI,0,sizeof(int)*ARRAYNUM);

	cudaStatus = cudaMalloc((void **)&dev_delaychandata,sizeof(int)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE/2));
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_delaychandata cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_delaychandata,0,sizeof(int)*ARRAYNUM*(FILTER_FRAME/DOWNSAMPLE/2));

	cudaStatus = cudaMalloc((void **)&dev_beamdata,sizeof(int)*MAXTRACETARNUM*(FILTER_FRAME/DOWNSAMPLE/2));
	if (cudaStatus != cudaSuccess)
	{
		printf ("dev_beamdata cudaMalloc Error! \n ");
	}
	cudaMemset((void **)&dev_beamdata,0,sizeof(int)*MAXTRACETARNUM*(FILTER_FRAME/DOWNSAMPLE/2));
	

	sourcedata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE)*sizeof(float));
	memset(sourcedata,0,(FILTER_FRAME/DOWNSAMPLE)*sizeof(float));

	shiftdata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE)*sizeof(float));
	memset(shiftdata,0,(FILTER_FRAME/DOWNSAMPLE)*sizeof(float));

	delayfilteroutdata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE+2*M)*sizeof(float));
	memset(delayfilteroutdata,0,(FILTER_FRAME/DOWNSAMPLE+2*M)*sizeof(float));	

	beamdata = (float *)malloc((FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float));
	memset(beamdata,0,(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float));

    cudaStatus = cudaMalloc((void **)&dev_tracebeam,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);

    cudaStatus = cudaMalloc((void **)&dev_tracebeam_spec,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_spec cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_spec,0,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);

    cudaStatus = cudaMalloc((void **)&dev_tracebeam_psd,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM*PSD_AVG_NUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_psd cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_psd,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM*PSD_AVG_NUM);

	cudaStatus = cudaMalloc((void **)&dev_tracebeam_psd_avg,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_psd_avg cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_psd_avg,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);

    cudaStatus = cudaMalloc((void **)&dev_tracebeam_demonspec,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_demonspec cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_demonspec,0,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);

    cudaStatus = cudaMalloc((void **)&dev_tracebeam_demonspec_band,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_demonspec_band cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_demonspec_band,0,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

    cudaStatus = cudaMalloc((void **)&dev_tracebeam_demon,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_demon cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_demon,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);

    cudaStatus = cudaMalloc((void **)&dev_tracebeam_demon_band_data,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_demon_band_data cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_demon_band_data,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

    trace_beam_psd = (float *)malloc(PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));
    memset(trace_beam_psd,0,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));

	trace_beam_psd_smooth = (float *)malloc(PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));
    memset(trace_beam_psd_smooth,0,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));

	cudaStatus = cudaMalloc((void **)&dev_tracebeam_psd_S,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_psd_S cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_psd_S,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);

	cudaStatus = cudaMalloc((void **)&dev_tracebeam_psd_E,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_psd_E cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_psd_E,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);	

	trace_beam_demon = (float *)malloc(PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float));
    memset(trace_beam_demon,0,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float));

	trace_beam_demon_smooth = (float *)malloc(DEM_RST_LEN*sizeof(float));
    memset(trace_beam_demon_smooth,0,DEM_RST_LEN*sizeof(float));

	cudaStatus = cudaMalloc((void **)&dev_tracebeam_demon_S,sizeof(cufftReal)*DEM_RST_LEN);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_demon_S cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_demon_S,0,sizeof(cufftReal)*DEM_RST_LEN);	

	cudaStatus = cudaMalloc((void **)&dev_tracebeam_demon_E,sizeof(cufftReal)*DEM_RST_LEN);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_tracebeam_demon_E cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_tracebeam_demon_E,0,sizeof(cufftReal)*DEM_RST_LEN);	

	cudaStatus = cudaMalloc((void **)&dev_trace_beam_demon_cut,sizeof(cufftReal)*DEM_RST_LEN);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_trace_beam_demon_cut cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_trace_beam_demon_cut,0,sizeof(cufftReal)*DEM_RST_LEN);	

	//------------------------------矢量通道变量------------------------------------------------
	cudaStatus = cudaMalloc((void **)&dev_vector_p_buf,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_p_buf cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_p_buf,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

	cudaStatus = cudaMalloc((void **)&dev_vector_x_buf,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_x_buf cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_x_buf,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

	cudaStatus = cudaMalloc((void **)&dev_vector_y_buf,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_y_buf cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_y_buf,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

    cudaStatus = cudaMalloc((void **)&dev_vector_p_psd,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*PSD_AVG_NUM);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_p_psd cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_p_psd,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*PSD_AVG_NUM);

    cudaStatus = cudaMalloc((void **)&dev_vector_p_spec,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_p_spec cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_p_spec,0,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

    cudaStatus = cudaMalloc((void **)&dev_vector_x_spec,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_x_spec cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_x_spec,0,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

    cudaStatus = cudaMalloc((void **)&dev_vector_y_spec,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_y_spec cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_y_spec,0,sizeof(cufftComplex)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));

    cudaStatus = cudaMalloc((void **)&dev_vector_psd_avg,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_psd_avg cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_psd_avg,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));	

	vector_p_psd = (float*)malloc(PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));
	memset(vector_p_psd,0,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));

	vector_p_psd_smooth = (float*)malloc(PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));
	memset(vector_p_psd_smooth,0,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float));

	cudaStatus = cudaMalloc((void **)&dev_vector_p_psd_S,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_p_psd_S cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_p_psd_S,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);

	cudaStatus = cudaMalloc((void **)&dev_vector_p_psd_E,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);
    if (cudaStatus != cudaSuccess)
    {
        printf ("dev_vector_p_psd_E cudaMalloc Error! \n ");
    }
    cudaMemset((void **)&dev_vector_p_psd_E,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);	

	//--------------------------------------------------------------------------------------------
	fDf=FS/DOWNSAMPLE*1.0/(PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
	idx1=(int)(10/fDf);
	idx2=(int)(5000/fDf);
	idxLen=idx2-idx1+1;

    DemFreqBandNum  = 4;
    DemStartFreq[0] = 2000.0; 
    DemEndFreq[0]   = 4000.0;

    DemStartFreq[1] = 4000.0; 
    DemEndFreq[1]   = 6000.0;

	DemStartFreq[2] = 6000.0; 
    DemEndFreq[2]   = 8000.0;

    DemStartFreq[3] = 8000.0; 
    DemEndFreq[3]   = 10000.0;
    //生成滑动窗正交多项式拟合时所用规范正交向量
	CalSmoothPara(&fSmoothA[0][0]);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	while(1)
	{
		retval = WaitForSingleObject(g_hFrameDataReadyEnvent,2000);
		FrameNum++;
		
		if(retval<0)
		{
			printf("Timeout!\n");
			return;
		}

		//移动缓冲区
		if(BUF_FLAG == 0)
		{
			for(int ii=0;ii<CHANNUM*2;ii++)
			{
				memmove(DownSamplingDataBufA+ii*FILTER_FRAME,DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,FRAMELEN*sizeof(float));
				memcpy(DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,ChannDataBufA+ii*FRAMELEN,FRAMELEN*sizeof(float));
			}
			cudaMemcpy(dev_x,DownSamplingDataBufA,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2,cudaMemcpyHostToDevice);
			BUF_FLAG = 1;
		}
		else
		{
			for(int ii=0;ii<CHANNUM*2;ii++)
			{
				memmove(DownSamplingDataBufA+ii*FILTER_FRAME,DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,FRAMELEN*sizeof(float));
				memcpy(DownSamplingDataBufA+ii*FILTER_FRAME+FRAMELEN,ChannDataBufB+ii*FRAMELEN,FRAMELEN*sizeof(float));
			}
			cudaMemcpy(dev_x,DownSamplingDataBufA,sizeof(cufftReal)*FILTER_FRAME*CHANNUM*2,cudaMemcpyHostToDevice);
			BUF_FLAG = 0;
		}
		
		cudaEventRecord(start1,NULL);

		//-----------------------------------------(1) 信号滤波降采样---------------------------------------------------
		//4.7ms
		for(int jj=0;jj<CHANNUM*2;jj++)
		{
			cufftExecR2C(Xplan,(cufftReal *)&dev_x[jj*FILTER_FRAME],(cufftComplex *)&dev_fft_x[jj*FILTER_FRAME]);
		}
		
		//频域相乘(13ms)
		DownSamplingFilter<<<CHANNUM*2*(FILTER_FRAME/2/THREADNUMPERBLK),THREADNUMPERBLK>>>(dev_fft_x,dev_fft_h,dev_fft_y,FILTER_FRAME);
	
		QueryPerformanceCounter(&nBeginTime); 
		//反变换(105ms)
		for(int jj=0;jj<CHANNUM*2;jj++)
		{
			cufftExecC2R(Yplan,(cufftComplex *)&dev_fft_y[jj*FILTER_FRAME],(cufftReal *)&dev_y[jj*FILTER_FRAME]);
			cudaMemcpy(dev_chanbuff+jj*FILTER_FRAME/DOWNSAMPLE,dev_chanbuff+jj*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2,FILTER_FRAME/DOWNSAMPLE/2*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		IFFTNormalize<<<CHANNUM*2*(FILTER_FRAME/2/THREADNUMPERBLK),THREADNUMPERBLK>>>(dev_y,dev_chanbuff,FILTER_FRAME);	

		QueryPerformanceCounter(&nEndTime);
		//-----------------------------------------(1) 信号滤波降采样结束---------------------------------------------------


		//-----------------------------------------(2) 频域波束形成---------------------------------------------------

		//使用缓冲区中的后FILTER_FRAME/DOWNSAMPLE/2点数据做频域波束形成，估计方位
		for (int ii=0;ii<ARRAYNUM;ii++)		
		{		
			cufftExecR2C(Beamplan,(cufftReal *)&dev_chanbuff[ii*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2],(cufftComplex *)&dev_fft[ii*(NFFT/2+1)]);
		}

		FD_Beamform<<<BlockRowNum*BEAMNUM,THREADNUMPERBLK>>>(dev_fft,dev_energy,PhiArray,nfl,nfh);//波束形成
		MatrixSumRow<<<BEAMNUM,1>>>(dev_energy,sum_energy,BlockRowNum,BEAMNUM);
		
		cudaMemcpy(c,sum_energy,BEAMNUM*sizeof(float),cudaMemcpyDeviceToHost);
		fwrite(c,sizeof(float),BEAMNUM,fp);
		//-----------------------------------------(2) 频域波束形成结束-----------------------------------------------


		//-----------------------------------------(3) 波束能量检测------------------------------------------
		//波束能量检测与跟踪
		memset(peak,0,BEAMNUM*sizeof(int));
		memset(valley,0,BEAMNUM*sizeof(int));
		findpeak(c,peak,BEAMNUM);
		findvalley(c,valley,BEAMNUM);
		bool targetexist = false;
		//memmove(beammatrix+BEAMNUM,beammatrix,4*BEAMNUM*sizeof(int));
		memset(pretracedtarget,0,sizeof(float)*BEAMNUM);
		memset(pretracedtargetIdx,0,sizeof(int)*BEAMNUM);
		pretracedtargetNum = 0;

		for(int kk=0;kk<BEAMNUM;kk++)
		{
			if(peak[kk] == 1)
			{
				//判断是否已跟踪该波束附近目标
				int jj=0;
				for(jj=0;jj<MAXTRACETARNUM;jj++)
				{
					//先找是否已跟踪
					if(abs(tracedtargetbeam[jj][0]-kk)<6 && tracedtargetbeam[jj][0]>0)   //已跟踪该目标，更新跟踪器角度
					{
						break;
					}
				}
				if(jj==MAXTRACETARNUM)  //未跟踪
				{
					targetexist = peakdetection(kk,c,valley,2.0);
				}
				else  //已跟踪，降低检测门限
				{
					targetexist = peakdetection(kk,c,valley,1.2);
				}
				if(targetexist)
				{
					pretracedtarget[pretracedtargetNum] = c[kk];
					pretracedtargetIdx[pretracedtargetNum] = kk;
					pretracedtargetNum++;
				}
			}
		}
		rbub(pretracedtarget,pretracedtargetIdx,BEAMNUM);

		if(FrameNum == 115)
		{
			FrameNum = FrameNum;
		}
		for(int kk=0;kk<pretracedtargetNum;kk++)
		{
			int jj=0;
			for(jj=0;jj<MAXTRACETARNUM;jj++)
			{
				//先找是否已跟踪
				if(abs(tracedtargetbeam[jj][0]-pretracedtargetIdx[kk])<6 && tracedtargetbeam[jj][0]>0)   //已跟踪该目标，更新跟踪器角度
				{
					tracedtargetbeam[jj][0] = pretracedtargetIdx[kk];
					tracedtargetbeam[jj][1] = FrameNum;
					break;
				}
			}

			if(jj==MAXTRACETARNUM)  //未跟踪该目标，找一个空的跟踪器跟踪
			{
				int ii = 0;
				for(ii=0;ii<MAXTRACETARNUM;ii++)
				{
					//先找是否已跟踪
					if(tracedtargetbeam[ii][0] < 0)
					{
						break;
					}
				}
				if(ii < MAXTRACETARNUM)           //有空置跟踪器
				{
					tracedtargetbeam[ii][0] = pretracedtargetIdx[kk];
					tracedtargetbeam[ii][1] = FrameNum;
				}
			}
		}
		//跟踪器管理，清空多帧未更新的跟踪器
		for(int jj=0;jj<MAXTRACETARNUM;jj++)
		{
			if(tracedtargetbeam[jj][0] >0 && FrameNum - tracedtargetbeam[jj][1] >= 5)
			{
				tracedtargetbeam[jj][0] = -1;
				tracedtargetbeam[jj][1] = -1;
				tracedtargetangle[jj] = -1.0f;
			}
		}
		//-----------------------------------------(3) 波束能量检测-------------------------------------


		//-----------------------------------------(4) 波束跟踪、跟踪波束 ------------------------------
	    cudaMemset((void **)&dev_tracebeam_demon,0,sizeof(cufftReal)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*MAXTRACETARNUM);
		for(int jj = 0;jj<MAXTRACETARNUM;jj++)
		{
			if(tracedtargetbeam[jj][0] >0)   //有跟踪目标
			{
				//波束内插
				i0 = tracedtargetbeam[jj][0]-1;
				i1 = tracedtargetbeam[jj][0];
				i2 = tracedtargetbeam[jj][0]+1;
				r0 = c[i0];
				r1 = c[i1];
				r2 = c[i2];
				delta_index = (r2-r0)/(4*r1-2*r0-2*r2);
				tracedtargetangle[jj] = (i1+delta_index)*180.0/BEAMNUM;
				DelayFilterGen<<<ARRAYNUM,2*M+1>>>(dev_delayFilter,M,tracedtargetangle[jj],dev_tau,dev_dI);
				//DelayFilterGen<<<ARRAYNUM,2*M+1>>>(dev_delayFilter,M,60.292690,dev_tau,dev_dI);
				cudaMemcpy(delayfiltercoff,dev_delayFilter,sizeof(cufftReal)*ARRAYNUM*(2*M+1),cudaMemcpyDeviceToHost);
				cudaMemcpy(delaytau,dev_tau,sizeof(cufftReal)*ARRAYNUM,cudaMemcpyDeviceToHost);
				cudaMemcpy(delaydI,dev_dI,sizeof(int)*ARRAYNUM,cudaMemcpyDeviceToHost);
				
				for(int kk = 0;kk<ARRAYNUM;kk++)
				{
					if(delaydI[kk] >= 0)
					{
						cudaMemcpy(dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE)+delaydI[kk],dev_chanbuff+kk*(FILTER_FRAME/DOWNSAMPLE),sizeof(cufftReal)*((FILTER_FRAME/DOWNSAMPLE)-delaydI[kk]),cudaMemcpyDeviceToDevice);
					}
					else
					{
						cudaMemcpy(dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE),dev_chanbuff+kk*(FILTER_FRAME/DOWNSAMPLE)-delaydI[kk],sizeof(cufftReal)*((FILTER_FRAME/DOWNSAMPLE)+delaydI[kk]),cudaMemcpyDeviceToDevice);
					}

					//cudaMemcpy(sourcedata,dev_chanbuff+kk*(FILTER_FRAME/DOWNSAMPLE),(FILTER_FRAME/DOWNSAMPLE)*sizeof(float),cudaMemcpyDeviceToHost);
					//cudaMemcpy(shiftdata,dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE),(FILTER_FRAME/DOWNSAMPLE)*sizeof(float),cudaMemcpyDeviceToHost);

					if(fabs(delaytau[kk]) > 0.0001)
					{
						FineDelayFilter<<<(FILTER_FRAME/DOWNSAMPLE+2*M),2*M+1>>>((cufftReal *)&dev_delayfilterbuf[kk*FILTER_FRAME/DOWNSAMPLE],(cufftReal *)&dev_delayfilterout[kk*(FILTER_FRAME/DOWNSAMPLE+2*M)],(cufftReal *)&dev_delayFilter[kk*(2*M+1)],M);
					}
					else
					{
						cudaMemcpy(dev_delayfilterout+kk*(FILTER_FRAME/DOWNSAMPLE+2*M)+M,dev_delayfilterbuf+kk*(FILTER_FRAME/DOWNSAMPLE),sizeof(cufftReal)*(FILTER_FRAME/DOWNSAMPLE),cudaMemcpyDeviceToDevice);
					}
					cudaMemcpy(dev_delaychandata+kk*(FILTER_FRAME/DOWNSAMPLE/2),dev_delayfilterout+kk*(FILTER_FRAME/DOWNSAMPLE+2*M)+M+FILTER_FRAME/DOWNSAMPLE/4,sizeof(cufftReal)*FILTER_FRAME/DOWNSAMPLE/2,cudaMemcpyDeviceToDevice);
				}
			
				MatrixSumRow<<<FILTER_FRAME/DOWNSAMPLE/2,1>>>(dev_delaychandata,dev_beamdata+jj*FILTER_FRAME/DOWNSAMPLE/2,ARRAYNUM,FILTER_FRAME/DOWNSAMPLE/2);
				cudaMemcpy(beamdata,dev_beamdata+jj*FILTER_FRAME/DOWNSAMPLE/2,FILTER_FRAME/DOWNSAMPLE/2*sizeof(float),cudaMemcpyDeviceToHost);
                //fwrite(beamdata,sizeof(float),FILTER_FRAME/DOWNSAMPLE/2,fpbeam);

				//功率谱
                cudaMemcpy(dev_tracebeam+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),dev_tracebeam+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)+(FILTER_FRAME/DOWNSAMPLE/2),(FILTER_FRAME/DOWNSAMPLE/2)*(PSD_LEN-1)*sizeof(cufftReal),cudaMemcpyDeviceToDevice);
                cudaMemcpy(dev_tracebeam+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)+(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2),dev_beamdata+jj*FILTER_FRAME/DOWNSAMPLE/2,(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(cufftReal),cudaMemcpyDeviceToDevice);
				//功率谱缓冲区移位
				cudaMemcpy(dev_tracebeam_psd+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*PSD_AVG_NUM,dev_tracebeam_psd+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*PSD_AVG_NUM+PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),(PSD_AVG_NUM-1)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(cufftReal),cudaMemcpyDeviceToDevice);
                cufftExecR2C(PSDplan,(cufftReal *)&dev_tracebeam[jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)],(cufftComplex *)&dev_tracebeam_spec[jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)]);
                Psd<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_tracebeam_spec+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),dev_tracebeam_psd+jj*PSD_LEN*PSD_AVG_NUM*(FILTER_FRAME/DOWNSAMPLE/2)+(PSD_AVG_NUM-1)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);
				//功率谱平均
				PsdAverage<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_tracebeam_psd+jj*PSD_LEN*PSD_AVG_NUM*(FILTER_FRAME/DOWNSAMPLE/2),dev_tracebeam_psd_avg+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
                cudaMemcpy(trace_beam_psd,dev_tracebeam_psd_avg+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float),cudaMemcpyDeviceToHost);
                //fwrite(trace_beam_psd,sizeof(float),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2,fpbeam);
				//功率谱平滑
			    MySmooth(trace_beam_psd+idx1, idxLen, &fSmoothA[0][0], SMOOTH_N, 3, SMOOTH_N, 5, trace_beam_psd_smooth+idx1);
				MySmooth(trace_beam_psd_smooth+idx1, idxLen, &fSmoothA[0][0], SMOOTH_N, 2, SMOOTH_N, 5, trace_beam_psd_smooth+idx1);
				cudaMemcpy(dev_tracebeam_psd_S,trace_beam_psd_smooth,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float),cudaMemcpyHostToDevice);
				//计算差值谱
				PsdSub<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_tracebeam_psd_avg+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),dev_tracebeam_psd_S,dev_tracebeam_psd_E,idx1,idx2);
				cudaMemcpy(trace_beam_psd_smooth,dev_tracebeam_psd_E,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float),cudaMemcpyDeviceToHost);
				//fwrite(trace_beam_psd_smooth,sizeof(float),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2,fpbeam);
				//计算差值谱方差
				fPsdEVar=0.0;
				for (int ii=idx1;ii<=idx2;ii++)
				{
					fPsdEVar+=trace_beam_psd_smooth[ii]*trace_beam_psd_smooth[ii];
				}
				fPsdEVar/=(float)(idx2-idx1+1);
				fPsdEVar=sqrtf(fPsdEVar);


				//解调谱
				for(int ii =0;ii<DemFreqBandNum;ii++)
				{
					cudaMemcpy(dev_tracebeam_demonspec_band,dev_tracebeam_spec+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(cufftComplex),cudaMemcpyDeviceToDevice);
					//频域带通滤波
					FrequencyDomainFilter<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_tracebeam_demonspec_band,fDf,DemStartFreq[ii],DemEndFreq[ii]);
					cufftExecC2R(DEMONBandplan,dev_tracebeam_demonspec_band,dev_tracebeam_demon_band_data);
					SignalSqr<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_tracebeam_demon_band_data);				
					cufftExecR2C(DEMONplan,dev_tracebeam_demon_band_data,dev_tracebeam_demonspec);
					DemonAdd<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_tracebeam_demonspec,dev_tracebeam_demon+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2), PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2));
				}
				cudaMemcpy(trace_beam_demon,dev_tracebeam_demon+jj*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),DEM_RST_LEN*sizeof(float),cudaMemcpyDeviceToHost);
				//前四个点赋值
				for(int ii=0;ii<6;ii++)
				{
					trace_beam_demon[ii] = trace_beam_demon[6];
				}
				//fwrite(trace_beam_demon,sizeof(float),DEM_RST_LEN,fpbeam);
				//解调谱平滑
			    MySmooth(trace_beam_demon, DEM_RST_LEN, &fSmoothA[0][0], SMOOTH_N, 3, SMOOTH_N, 5, trace_beam_demon_smooth);
				MySmooth(trace_beam_demon_smooth, DEM_RST_LEN, &fSmoothA[0][0], SMOOTH_N, 2, SMOOTH_N, 5, trace_beam_demon_smooth);
				//fwrite(trace_beam_demon_smooth,sizeof(float),DEM_RST_LEN,fpbeam);
				cudaMemcpy(dev_trace_beam_demon_cut,trace_beam_demon,DEM_RST_LEN*sizeof(cufftReal),cudaMemcpyHostToDevice);
				cudaMemcpy(dev_tracebeam_demon_S,trace_beam_demon_smooth,DEM_RST_LEN*sizeof(cufftReal),cudaMemcpyHostToDevice);
				//fwrite(trace_beam_demon_smooth,sizeof(float),DEM_RST_LEN,fpbeam)
				DemonSub<<<DEM_RST_LEN,1>>>(dev_trace_beam_demon_cut,dev_tracebeam_demon_S,dev_tracebeam_demon_E);
				cudaMemcpy(trace_beam_demon_smooth,dev_tracebeam_demon_E,DEM_RST_LEN*sizeof(cufftReal),cudaMemcpyDeviceToHost);
				//fwrite(trace_beam_demon_smooth,sizeof(float),DEM_RST_LEN,fpbeam);
				fDemonEVar=0.0;
				for (int ii=0;ii<DEM_RST_LEN;ii++)
				{
					fDemonEVar+=trace_beam_demon_smooth[ii]*trace_beam_demon_smooth[ii];
				}
				fDemonEVar/=(float)(DEM_RST_LEN);
				fDemonEVar=sqrtf(fDemonEVar);

				//线谱提取
				int ll = 0;
				if(FrameNum >= 8)
				{
					nPlineNum = 0;
					memset(fPlineInfo,0,MAXTRACETARNUM*LINE_NUM*4*sizeof(float));
					for(int ii=idx1;ii<=idx2;ii++)
					{
						if(trace_beam_psd_smooth[ii]>4.0*fPsdEVar && trace_beam_psd_smooth[ii]>trace_beam_psd_smooth[ii-1] && trace_beam_psd_smooth[ii]>trace_beam_psd_smooth[ii+1] )
						{
							if(nPlineNum<LINE_NUM)
							{
								//线谱归并
								for(ll = 0;ll<nPlineNum;ll++)
								{
									if(fabs(fPlineInfo[jj][ll][1]-(float)ii*fDf)<1.0)
									{
										break;
									}
								}
								if(ll == nPlineNum)
								{
									fPlineInfo[jj][nPlineNum][0] = trace_beam_psd_smooth[ii];    //信噪比
									fPlineInfo[jj][nPlineNum][1] = (float)ii*fDf;                //线谱信噪比
									fPlineInfo[jj][nPlineNum][2] = trace_beam_psd[ii];
									fPlineInfo[jj][nPlineNum][3] = tracedtargetangle[jj];
									if(fPlineInfo[jj][nPlineNum][3] > 180.0)
									{
										fPlineInfo[jj][nPlineNum][3] -= 360.0;
									}
									else if(fPlineInfo[jj][nPlineNum][3] < -180.0)
									{
										fPlineInfo[jj][nPlineNum][3] += 360.0;
									}
									nPlineNum++;
								}
								else if(trace_beam_psd_smooth[ii] > fPlineInfo[jj][ll][0])
								{
									fPlineInfo[jj][ll][0] = trace_beam_psd_smooth[ii];
									fPlineInfo[jj][ll][1] = (float)ii*fDf;
									fPlineInfo[jj][ll][2] = trace_beam_psd[ii];
									fPlineInfo[jj][ll][3] = tracedtargetangle[jj];;
									if(fPlineInfo[jj][ll][3] > 180.0)
									{
										fPlineInfo[jj][ll][3] -= 360.0;
									}
									else if(fPlineInfo[jj][ll][3] < -180.0)
									{
										fPlineInfo[jj][ll][3] += 360.0;
									}		
								}
							}
						}
					}
					nDlineNum = 0;
					memset(fDlineInfo,0,MAXTRACETARNUM*LINE_NUM*2*sizeof(float));
					for(int ii = 4;ii<DEM_RST_LEN-1;ii++)
					{
						if(trace_beam_demon_smooth[ii]>6.0*fDemonEVar && trace_beam_demon_smooth[ii]>trace_beam_demon_smooth[ii-1] && trace_beam_demon_smooth[ii]>trace_beam_demon_smooth[ii+1])
						{
							if(nDlineNum<LINE_NUM)
							{
								fDlineInfo[jj][nDlineNum][0]=trace_beam_demon_smooth[jj];
								fDlineInfo[jj][nDlineNum][1]=ii*fDf;
								nDlineNum++;
							}
						}
					}
					//for(int ii = 0;ii<nDlineNum;ii++)
					//{
					//	printf("%d:%.3f\n",ii+1,fDlineInfo[jj][ii][1]);
					//}
				}
			}
		}

		//--------------------------矢量处理----------------------------------------------------
		cudaMemcpy(dev_vector_p_buf,dev_vector_p_buf+(FILTER_FRAME/DOWNSAMPLE/2),(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_vector_p_buf+(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2),dev_chanbuff+VECTOR_P_IDX*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2,(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_vector_x_buf,dev_vector_x_buf+(FILTER_FRAME/DOWNSAMPLE/2),(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_vector_x_buf+(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2),dev_chanbuff+VECTOR_X_IDX*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2,(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_vector_y_buf,dev_vector_y_buf+(FILTER_FRAME/DOWNSAMPLE/2),(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_vector_y_buf+(PSD_LEN-1)*(FILTER_FRAME/DOWNSAMPLE/2),dev_chanbuff+VECTOR_Y_IDX*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2,(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(float),cudaMemcpyDeviceToDevice);

		cufftExecR2C(PSDplan,(cufftReal *)&dev_vector_p_buf[0],(cufftComplex *)&dev_vector_p_spec[0]);
		cufftExecR2C(PSDplan,(cufftReal *)&dev_vector_x_buf[0],(cufftComplex *)&dev_vector_x_spec[0]);
		cufftExecR2C(PSDplan,(cufftReal *)&dev_vector_y_buf[0],(cufftComplex *)&dev_vector_y_spec[0]);

		cudaMemcpy(dev_vector_p_psd,dev_vector_p_psd+PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),(PSD_AVG_NUM-1)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)*sizeof(cufftReal),cudaMemcpyDeviceToDevice);
        Psd<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_vector_p_spec,dev_vector_p_psd+(PSD_AVG_NUM-1)*PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2);
		PsdAverage<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_vector_p_psd,dev_vector_psd_avg);
		cudaMemcpy(vector_p_psd,dev_vector_psd_avg,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float),cudaMemcpyDeviceToHost);
		fwrite(vector_p_psd,sizeof(float),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2,fpbeam);

		MySmooth(vector_p_psd+idx1, idxLen, &fSmoothA[0][0], SMOOTH_N, 3, SMOOTH_N, 5, vector_p_psd_smooth+idx1);
		MySmooth(vector_p_psd_smooth+idx1, idxLen, &fSmoothA[0][0], SMOOTH_N, 2, SMOOTH_N, 5, vector_p_psd_smooth+idx1);
		cudaMemcpy(dev_vector_p_psd_S,vector_p_psd_smooth,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float),cudaMemcpyHostToDevice);
		PsdSub<<<PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_vector_psd_avg,dev_vector_p_psd_S,dev_vector_p_psd_E,idx1,idx2);
		cudaMemcpy(vector_p_psd_smooth,dev_vector_p_psd_E,PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2*sizeof(float),cudaMemcpyDeviceToHost);
		//fwrite(vector_p_psd_smooth,sizeof(float),PSD_LEN*(FILTER_FRAME/DOWNSAMPLE/2)/2,fpbeam);
		fVectorPsdEVar=0.0;
		for (int ii=idx1;ii<=idx2;ii++)
		{
			fVectorPsdEVar+=vector_p_psd_smooth[ii]*vector_p_psd_smooth[ii];
		}
		fVectorPsdEVar/=(float)(idx2-idx1+1);
		fVectorPsdEVar=sqrtf(fVectorPsdEVar);
		//线谱提取
		int ll = 0;
		if(FrameNum >= 8)
		{
			nVectorPlineNum = 0;
			memset(fVectorPlineInfo,0,LINE_NUM*4*sizeof(float));
			for(int ii=idx1;ii<=idx2;ii++)
			{
				if(vector_p_psd_smooth[ii]>4.0*fVectorPsdEVar && vector_p_psd_smooth[ii]>vector_p_psd_smooth[ii-1] && vector_p_psd_smooth[ii]>vector_p_psd_smooth[ii+1] )
				{
					if(nVectorPlineNum<LINE_NUM)
					{
						//线谱归并
						for(ll = 0;ll<nVectorPlineNum;ll++)
						{
							if(fabs(fVectorPlineInfo[ll][1]-(float)ii*fDf)<1.0)
							{
								break;
							}
						}
						if(ll == nVectorPlineNum)
						{
							fVectorPlineInfo[nVectorPlineNum][0] = vector_p_psd_smooth[ii];    //信噪比
							fVectorPlineInfo[nVectorPlineNum][1] = (float)ii*fDf;                //线谱信噪比
							fVectorPlineInfo[nVectorPlineNum][2] = vector_p_psd[ii];
							//fVectorPlineInfo[nVectorPlineNum][3] = tracedtargetangle[jj];
							cufftComplex P_f,Vx_f,Vy_f;
							cudaMemcpy(&P_f,dev_vector_p_spec+ii,sizeof(cufftComplex),cudaMemcpyDeviceToHost);
							cudaMemcpy(&Vx_f,dev_vector_x_spec+ii,sizeof(cufftComplex),cudaMemcpyDeviceToHost);
							cudaMemcpy(&Vy_f,dev_vector_y_spec+ii,sizeof(cufftComplex),cudaMemcpyDeviceToHost);

							if(FrameNum == 20)
							{
								FrameNum = FrameNum;
							}

							fVectorPlineInfo[nVectorPlineNum][3] = VectorThetSPF(P_f, Vx_f, Vy_f);
							if(fVectorPlineInfo[nVectorPlineNum][3] > 180.0)
							{
								fVectorPlineInfo[nVectorPlineNum][3] -= 360.0;
							}
							else if(fVectorPlineInfo[nVectorPlineNum][3] < -180.0)
							{
								fVectorPlineInfo[nVectorPlineNum][3] += 360.0;
							}
							nVectorPlineNum++;
						}
						else if(vector_p_psd_smooth[ii] > fVectorPlineInfo[ll][0])
						{
							fVectorPlineInfo[ll][0] = vector_p_psd_smooth[ii];
							fVectorPlineInfo[ll][1] = (float)ii*fDf;
							fVectorPlineInfo[ll][2] = vector_p_psd[ii];
							cufftComplex P_f,Vx_f,Vy_f;
							cudaMemcpy(&P_f,dev_vector_p_spec+ii,sizeof(cufftComplex),cudaMemcpyDeviceToHost);
							cudaMemcpy(&Vx_f,dev_vector_x_spec+ii,sizeof(cufftComplex),cudaMemcpyDeviceToHost);
							cudaMemcpy(&Vy_f,dev_vector_y_spec+ii,sizeof(cufftComplex),cudaMemcpyDeviceToHost);

							fVectorPlineInfo[nVectorPlineNum][3] = VectorThetSPF(P_f, Vx_f, Vy_f);
							if(fVectorPlineInfo[ll][3] > 180.0)
							{
								fVectorPlineInfo[ll][3] -= 360.0;
							}
							else if(fVectorPlineInfo[ll][3] < -180.0)
							{
								fVectorPlineInfo[ll][3] += 360.0;
							}		
						}
					}
				}
			}
		}
		for(int ii = 0;ii<nVectorPlineNum;ii++)
		{
			printf("fVectorPlineInfo %d:%.3f\n",ii+1,fVectorPlineInfo[ii][3]);
		}
		//--------------------------------------------------------------------------------------
		cudaEventRecord(stop1,NULL);
		cudaEventSynchronize(stop1);
		//time=(double)(nEndTime.QuadPart-nBeginTime.QuadPart)/(double)nFreq.QuadPart;
		cudaEventElapsedTime(&msecTotal,start1,stop1);
		printf("%d:%f;%d,%d;%d,%d;%d,%d\n",FrameNum,msecTotal,tracedtargetbeam[0][0],tracedtargetbeam[0][1],tracedtargetbeam[1][0],tracedtargetbeam[1][1],tracedtargetbeam[2][0],tracedtargetbeam[2][1]);
		fprintf(fplog,"%d:%f;%d,%d;%d,%d;%d,%d\n",FrameNum,msecTotal,tracedtargetbeam[0][0],tracedtargetbeam[0][1],tracedtargetbeam[1][0],tracedtargetbeam[1][1],tracedtargetbeam[2][0],tracedtargetbeam[2][1]);
		fflush(fplog);
	}
	fclose(fp);
	fp = NULL;
	fclose(fplog);
	fplog = NULL;
	fclose(fpbeam);
	fpbeam = NULL;
}
int  fir1(int n,int band,float fl,float fh,float fs,int wn, float *h)
{
	int i,n2,mid;
	float sum = 0;
	float s,wc1,wc2,beta = 0,delay;
	float fln = fl / fs;
	float fhn = fh / fs;

	beta = 6;
	if((n%2)==0)
	{
		n2=n/2-1;
		mid=1;
	}
	else
	{
		n2=n/2;
		mid=0;
	}
	delay=n/2.0;
	wc1=2.0*PI*fln;
	if(band>=3) wc2=2.0*PI*fhn;
	switch(band)
	{
	case 1://低通
		{
			for (i=0;i<=n2;i++)
			{
				s=i-delay;
				*(h+i)=(sin(wc1*s)/(PI*s))*window(wn,n+1,i,beta);
				*(h+n-i)=*(h+i);
			}
			if(mid==1) *(h+n/2)=wc1/PI;
			for(i=0;i<=n;i++)
			{
				sum=sum+*(h+i);
			}
			for(i=0;i<=n;i++)
			{
				*(h+i)=*(h+i)/fabs(sum);
			}
			break;
		}
	case 2: //高通
		{
			for (i=0;i<=n2;i++)
			{
				s=i-delay;
				*(h+i)=(sin(PI*s)-sin(wc1*s))/(PI*s);
				*(h+i)=*(h+i)*window(wn,n+1,i,beta);
				*(h+n-i)=*(h+i);
			}
			if(mid==1) *(h+n/2)=1.0-wc1/PI;
			break;
		}
	case 3: //带通
		{
			for (i=0;i<=n2;i++)
			{
				s=i-delay;
				*(h+i)=(sin(wc2*s)-sin(wc1*s))/(PI*s);
				*(h+i)=*(h+i)*window(wn,n+1,i,beta);
				*(h+n-i)=*(h+i);
			}
			if(mid==1) *(h+n/2)=(wc2-wc1)/PI;
			break;
		}
	case 4: //带阻
		{
			for (i=0;i<=n2;i++)
			{
				s=i-delay;
				*(h+i)=(sin(wc1*s)+sin(PI*s)-sin(wc2*s))/(PI*s);
				*(h+i)=*(h+i)*window(wn,n+1,i,beta);
				*(h+n-i)=*(h+i);
			}
			if(mid==1) *(h+n/2)=(wc1+PI-wc2)/PI;
			break;
		}
	}
	return 0;
}

float window(int type,int n,int i,float beta)
{
	int k;
	float w=1.0;
	switch(type)
	{
	case 1: //矩形窗
		{
			w=1.0;
			break;
		}
	case 2: //图基窗
		{
			k=(n-2)/10;
			if(i<=k) w=0.5*(1.0-cos(i*PI/(k+1)));
			if(i>n-k-2) w=0.5*(1.0-cos((n-i-1)*PI/(k+1)));
			break;
		}
	case 3: //三角窗
		{
			w=1.0-fabs(1.0-2*i/(n-1.0));
			break;
		}
	case 4: //汉宁窗
		{
			w=0.5*(1.0-cos(2*i*PI/(n-1.0)));
			break;
		}
	case 5: //海明窗
		{
			w=0.54-0.46*cos(2*i*PI/(n-1.0));
			break;
		}
	case 6: //布拉克曼窗
		{
			w=0.42-0.5*cos(2*i*PI/(n-1.0))+0.08*cos(4*i*PI/(n-1.0));
			break;
		}
	case 7: //凯塞窗
		{
			w=kaiser(i,n,beta);
			break;
		}
	}
	return(w);
}

float kaiser(int i,int n,float beta)  //凯塞窗，i为序号，n为滤波器长度
{
	float a,w,a2,b1,b2,beta1;
	b1=bessel0(beta);
	a=2.0*i/(float)(n-1)-1.0;
	a2=a*a;
	beta1=beta*sqrt(1.0-a2);
	b2=bessel0(beta1);
	w=b2/b1;
	return(w);
}

float bessel0(float x)  //零阶贝塞尔函数
{
	int i;
	float dd,y,d2,sum = 0;
	y=x/2.0;
	dd=1.0;
	for(i=1;i<=25;i++)
	{
		dd=dd*y/i;
		d2=dd*dd;
		sum=sum+d2;
		if(d2<sum*(1.0e-8)) break;
	}
	return(sum);
}

void findpeak(float *data, int *p,int dn)
{
	int acc=0,acc1=0;
	int i,j;
	float a0=0.0,a1=0.0;
	for(i=0;i<dn;i++)
	{
		a0=*(data+i);
		//先向前找
        for(j=1;j<11;j++)
		{
			if ((i+j)>=dn)
			{
				a1=*(data+i+j-dn);
			}
			else
			{
				a1=*(data+i+j);
			}
			if (a0>a1)
			{
				acc=acc+1;
			}
		}
        a0=*(data+i);
		////再向后找
        for(j=1;j<11;j++)
		{
			if ((i-j)<0)
			{
				a1=*(data+i-j+dn);
			}
			else
			{
				a1=*(data+i-j);
			}
			if (a0>a1)
			{
				acc1=acc1+1;
			}
		}
	  if ((acc==10) && (acc1==10))
	  {
         *(p+i)=1;
	  }
	  acc=0;
      acc1=0;
	}
}

void findvalley(float *data, int *p,int dn)
{
	int acc=0,acc1=0;
	int i,j;
	float a0=0.0,a1=0.0;
	for(i=0;i<dn;i++)
	{
		a0=*(data+i);
		//先向前找
        for(j=1;j<6;j++)
		{
			if ((i+j)>=dn)
			{
				//a1=*(data+i+j-dn);
				break;
			}
			else
			{
				a1=*(data+i+j);
			}
			if (a0<a1)
			{
				acc=acc+1;
			}
		}
		if(j<5)  //循环因break退出，到了波束号最大值
		{
			acc = 5; 
		}
        a0=*(data+i);
		////再向后找
        for(j=1;j<6;j++)
		{
			if ((i-j)<0)
			{
				//a1=*(data+i-j+dn);
				break;
			}
			else
			{
				a1=*(data+i-j);
			}
			if (a0<a1)
			{
				acc1=acc1+1;
			}
		}
		if(j<5)  //循环因break退出，到了波束号最小值
		{
			acc1 = 5; 
		}
		if ((acc==5) && (acc1==5))
		{
		    *(p+i)=1;
		}
		acc=0;
		acc1=0;
	}
}

bool peakdetection(int beamidx,float *be,int *valley,float threshold)
{
	int index = 0,ll=0;
	float pvr1 = 1.0,pvr2 = 1.0;
	if(beamidx >= STARTBEAM && beamidx <= ENDBEAM)
	{
		for(ll=beamidx+1;ll<BEAMNUM;ll++)
		{
			if(valley[ll] == 1)
			{
				index = ll;
				break;
			}
		}
		if(ll<=BEAMNUM-1)
		{
			pvr1 = be[beamidx] / be[index];
		}

		for(ll=beamidx-1;ll>=0;ll--)
		{
			if(valley[ll] == 1)
			{
				index = ll;
				break;
			}
		}
		if(ll>=0)
		{
			pvr2 = be[beamidx] / be[index];
		}

		if(pvr1 >= threshold && pvr2 >= threshold)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

void rbub(float *p,int *idx,int n)
{ 
	int m,k,j,i,xx;
    float dd;
    
	k=0; 
	m=n-1;
    while (k<m)
    { 
		j=m-1; m=0;
        for(i=k; i<=j; i++)
		{
			if(p[i]<p[i+1])
			{ 
				dd=p[i]; 
				p[i]=p[i+1]; 
				p[i+1]=dd; 
				xx = idx[i];
				idx[i] = idx[i+1];
				idx[i+1] = xx;
				m=i;
			}
		}
        j=k+1; 
		k=0;
        for (i=m; i>=j; i--)
		{
			if(p[i-1]<p[i])
			{ 
				dd=p[i]; 
				p[i]=p[i-1]; 
				p[i-1]=d; 
				xx = idx[i];
				idx[i] = idx[i-1];
				idx[i-1] = xx;
				k=i;
			}
		}
      }
    return;
  }

void MySmooth(float *datain,int nDataLen,float *paraA,int nParaLen,int nOrder,int nWindow,int nStep,float *dataout)
{
	int nFrameNum,ii,jj,nFrameCnt,idx;
	float rr[4]={0};
	float fsmooth_tmp[SMOOTH_N]={0};
	float fsmooth_tmp2[SMOOTH_N]={0};

	nFrameNum=(nDataLen-nWindow)/nStep+1;

	for (nFrameCnt=0;nFrameCnt<nFrameNum;nFrameCnt++)
	{
		if(nFrameCnt==0)
		{
			memcpy(fsmooth_tmp,datain,nWindow*sizeof(float));
		}
		else
		{
			memcpy(&fsmooth_tmp[nWindow-nStep],&datain[nWindow+(nFrameCnt-1)*nStep],nStep*sizeof(float));
		}

		for (ii=0;ii<nOrder;ii++)
		{
			rr[ii]=0.0;
			for (jj=0;jj<nWindow;jj++)
			{
				rr[ii]+=fsmooth_tmp[jj]*fSmoothA[ii][jj];
			}
		}

		memset(fsmooth_tmp2,0,SMOOTH_N*sizeof(float));
		for (ii=0;ii<nWindow;ii++)
		{
				for (jj=0;jj<nOrder;jj++)
				{
					fsmooth_tmp2[ii]+=rr[jj]*fSmoothA[jj][ii];
				}
		}

		memcpy(&dataout[nFrameCnt*nStep],fsmooth_tmp2,nStep*sizeof(float));
		memcpy(fsmooth_tmp,&fsmooth_tmp2[nStep],(nWindow-nStep)*sizeof(float));
	}//for (nFrameCnt=0;nFrameCnt<nFrameNum-1;nFrameCnt++)

	if ((nFrameNum*nStep+nWindow)-nDataLen<nStep)
	{
		idx=(nFrameNum*nStep+nWindow)-nDataLen;
		memcpy(fsmooth_tmp,&fsmooth_tmp2[nStep-idx],(nWindow-nStep+idx)*sizeof(float));
		memcpy(&fsmooth_tmp[nWindow-nStep+idx],&datain[nWindow+(nFrameNum-1)*nStep],(nStep-idx)*sizeof(float));

		for (ii=0;ii<nOrder;ii++)
		{
			rr[ii]=0.0;
			for (jj=0;jj<nWindow;jj++)
			{
				rr[ii]+=fsmooth_tmp[jj]*fSmoothA[ii][jj];
			}
		}

		memset(fsmooth_tmp2,0,SMOOTH_N*sizeof(float));
		for (ii=0;ii<nWindow;ii++)
		{
			for (jj=0;jj<nOrder;jj++)
			{
				fsmooth_tmp2[ii]+=rr[jj]*fSmoothA[jj][ii];
			}
		}

		memcpy(&dataout[nFrameNum*nStep],&fsmooth_tmp2[idx],(nWindow-idx)*sizeof(float));

	}
	else//if ((nFrameNum*nStep+nWindow)-nDataLen<nStep)
	{
		memcpy(&dataout[nFrameNum*nStep],&fsmooth_tmp2[nStep],(nWindow-nStep)*sizeof(float));

	}//if ((nFrameNum*nStep+nWindow)-nDataLen<nStep)
}

void CalSmoothPara(float *para)
{
	float fpara[4][SMOOTH_N];
	float ftmp,ftmp2,ftmp3;
	int ii,jj;

	ftmp=sqrtf((float)(SMOOTH_N));
	ftmp=1.0/ftmp;
	for (ii=0;ii<SMOOTH_N;ii++)
	{
		fpara[0][ii]=ftmp;
	}

	ftmp2=0;

	for (ii=0;ii<SMOOTH_N;ii++)
	{
		fpara[1][ii]=(float)(ii-(SMOOTH_N-1)/2);
		fpara[2][ii]=fpara[1][ii]*fpara[1][ii];
		ftmp2+=fpara[2][ii];
		fpara[3][ii]=fpara[2][ii]*fpara[1][ii];
	}
	ftmp=1.0/sqrtf(ftmp2);
	ftmp3=0;
	for (ii=0;ii<SMOOTH_N;ii++)
	{
		fpara[1][ii]=fpara[1][ii]*ftmp;
		ftmp3+=fpara[1][ii]*fpara[3][ii];
	}

	ftmp=0;
	ftmp2=ftmp2/(float)(SMOOTH_N);
	for (ii=0;ii<SMOOTH_N;ii++)
	{
		fpara[2][ii]=fpara[2][ii]-ftmp2;
		ftmp+=fpara[2][ii]*fpara[2][ii];
	}
	ftmp=1.0/sqrtf(ftmp);
	ftmp2=0;
	for (ii=0;ii<SMOOTH_N;ii++)
	{
		fpara[2][ii]=fpara[2][ii]*ftmp;
		fpara[3][ii]=fpara[3][ii]-ftmp3*fpara[1][ii];
		ftmp2+=fpara[3][ii]*fpara[3][ii];
	}
	ftmp=1.0/sqrtf(ftmp2);
	for (ii=0;ii<SMOOTH_N;ii++)
	{
		fpara[3][ii]=fpara[3][ii]*ftmp;
	}

	memcpy(para,&fpara[0][0],sizeof(float)*4*SMOOTH_N);
}
