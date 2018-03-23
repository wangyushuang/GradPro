// /home/ubuntu/Desktop/GPU/main.c
// nvcc main.cu -o test -lstdc++ -lpthread -lcufft -lpcap -std=c++11 -lpcap
#include <pcap.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/tcp.h>
#include <netinet/ip_icmp.h>
#include <net/ethernet.h>
#include <netinet/if_ether.h>
#include <netinet/ether.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <memory.h>
#include <malloc.h>
#include <iostream>
//--------------CUDA----------------
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <cufft.h>
//#include <cufftXt.h>

//-------------------------------------

// ----------------------------------------
#define     NFFT					 16384	      //
#define     PI				                 3.1415926f
#define     UWC						 1500.0f      //
#define     FS						 100000       //
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
#define     ONLINEMODE               0
#define     FILEMODE                 1
#define     DEST_PORT                0
#define     PSD_LEN                  20
#define     PSD_AVG_NUM              8
#define     EPS                      1e-8
#define     SMOOTH_N                 100
#define     LINE_NUM                 16
#define     DEM_RST_LEN              1024
#define     VECTOR_P_IDX             22
#define     VECTOR_X_IDX             16
#define     VECTOR_Y_IDX             18
// -----------------------------------------------------
void *ReadBoard1Data(void *lParam);
void *ReadBoard2Data(void *lParam);
void *DataFormatting(void *lParam);
void *ReceiveNetwork(void *lParam);
void *ArraySignalProcessing(void *lParam);
//-----------------------------------------------------
pthread_mutex_t count_lock_BoardDataReady;
pthread_mutex_t count_lock_Board1DataReady;
pthread_mutex_t count_lock_Board2DataReady;
pthread_mutex_t count_lock_FrameDataReady;
pthread_cond_t  cond_BoardDataReady;
pthread_cond_t  cond_Board1DataReady;
pthread_cond_t  cond_Board2DataReady;
pthread_cond_t  cond_FrameDataReady;
unsigned int    count_BoardDataReady;
unsigned int    count_Board1DataReady;
unsigned int    count_Board2DataReady;
unsigned int    count_FrameDataReady;
//-----------------------------------------------------
int *DataBufA_B1 = NULL;
int *DataBufB_B1 = NULL;
int *DataBufA_B2 = NULL;
int *DataBufB_B2 = NULL;
float *ChannDataBufA = NULL;
float *ChannDataBufB = NULL;
float *DownSamplingDataBufA = NULL;
float *DownSamplingDataBufB = NULL;
//---------------------------------------------------
int   fir1(int n,int band,float fl,float fh,float fs,int wn, float *h);
float window(int type,int n,int i,float beta);
float kaiser(int i,int n,float beta);
float bessel0(float x);
void  findpeak(float *data, int *p,int dn);
void  findvalley(float *data, int *p,int dn);
bool  peakdetection(int beamidx,float *be,int *valley,float threshold);
void  rbub(float *p,int *idx,int n);
void  MySmooth(float *datain,int nDataLen,float *paraA,int nParaLen,int nOrder,int nWindow,int nStep,float *dataout);
void  CalSmoothPara(float *para);
//-----------------------------------------------------
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
// -----------------------------------------------------------
int main()
{
	pthread_t t_ReceiveNetworkData;
	pthread_t t_DataFormatting;
	pthread_t t_ArraySignalProcessing;
        pthread_t t_ReadBoard1Data;
        pthread_t t_ReadBoard2Data;

	cond_BoardDataReady = PTHREAD_COND_INITIALIZER;
	cond_Board1DataReady = PTHREAD_COND_INITIALIZER;
	cond_Board2DataReady = PTHREAD_COND_INITIALIZER;
	cond_FrameDataReady = PTHREAD_COND_INITIALIZER;

	count_lock_BoardDataReady = PTHREAD_MUTEX_INITIALIZER;
	count_lock_Board1DataReady = PTHREAD_MUTEX_INITIALIZER;
	count_lock_Board2DataReady = PTHREAD_MUTEX_INITIALIZER;
	count_lock_FrameDataReady = PTHREAD_MUTEX_INITIALIZER;

	pthread_create(&t_ArraySignalProcessing,NULL,ArraySignalProcessing,(void *)NULL);
        pthread_create(&t_DataFormatting,NULL,DataFormatting,(void *)NULL);

#if ONLINEMODE
    pthread_create(&t_ReceiveNetworkData,NULL,ReceiveNetwork,(void *)NULL);
#endif

#if FILEMODE
    pthread_create(&t_ReadBoard1Data,NULL,ReadBoard1Data,(void *)NULL);
    pthread_create(&t_ReadBoard2Data,NULL,ReadBoard2Data,(void *)NULL);
#endif
	pthread_join(t_ArraySignalProcessing, NULL);
	return 0;
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
    case 1://
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
    case 2: //
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
    case 3: //
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
    case 4: //
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
    case 1: //
        {
            w=1.0;
            break;
        }
    case 2: //
        {
            k=(n-2)/10;
            if(i<=k) w=0.5*(1.0-cos(i*PI/(k+1)));
            if(i>n-k-2) w=0.5*(1.0-cos((n-i-1)*PI/(k+1)));
            break;
        }
    case 3: //
        {
            w=1.0-fabs(1.0-2*i/(n-1.0));
            break;
        }
    case 4: //
        {
            w=0.5*(1.0-cos(2*i*PI/(n-1.0)));
            break;
        }
    case 5: //
        {
            w=0.54-0.46*cos(2*i*PI/(n-1.0));
            break;
        }
    case 6: //
        {
            w=0.42-0.5*cos(2*i*PI/(n-1.0))+0.08*cos(4*i*PI/(n-1.0));
            break;
        }
    case 7: //
        {
            w=kaiser(i,n,beta);
            break;
        }
    }
    return(w);
}

float kaiser(int i,int n,float beta)  //
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

float bessel0(float x)  //
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

void findpeak(float *data, int *p,int dn)
{
    int acc=0,acc1=0;
    int i,j;
    float a0=0.0,a1=0.0;
    for(i=0;i<dn;i++)
    {
        a0=*(data+i);
        //
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
        //
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
        //
        for(j=1;j<6;j++)
        {
            if ((i+j)>=dn)
            {
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
        if(j<5)  //
        {
            acc = 5;
        }
        a0=*(data+i);
        //
        for(j=1;j<6;j++)
        {
            if ((i-j)<0)
            {
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
        if(j<5)  //
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

    //
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

    Mabs[tid]=tempX*tempX+tempY*tempY;

    //
    __syncthreads();

    //
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
    int bid = 0;
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

    //
    __syncthreads();

    k = tid-m;
    h[bid*(2*m+1)+tid] = sin(k*1.0*PI-tau[bid]*PI+0.000001)/(k*1.0*PI-tau[bid]*PI+0.000001);

    //
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

    //
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
}

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

void *ArraySignalProcessing(void *lParam)
{
    //int retval = -1;
    int BUF_FLAG = 0;
    int FrameNum = 0;

    //-----------------Downsampling filter-------------------------------
    float h[FIRORDER+1] = {0.0};
    float fl = 100.0f,fh = 10e3f;
    cudaError    cudaStatus;
    cufftReal    *dev_x=NULL;
    cufftReal    *dev_h=NULL;
    cufftComplex *dev_fft_x=NULL;
    cufftComplex *dev_fft_h=NULL;
    cufftComplex *dev_fft_y=NULL;
    cufftReal    *dev_y=NULL;
    cufftReal    *dev_chanbuff=NULL;
    float        *FilteredDataout = NULL;
    float        *DownSamplingData = NULL;
    cufftHandle  Hplan;
    cufftHandle  Xplan;
    cufftHandle  Yplan;
    //----------------------------------------------------------------
    //--------------------------Process Time Test---------------------
    cudaEvent_t start1;
    cudaEvent_t stop1;
    float msecTotal = 0.0f;
    //----------------------------------------------------------------
    //--------------------------Beamforming and Tracing---------------
    int nfl = (int)((2000.0/(FS/DOWNSAMPLE)*NFFT)+0.5);
    int nfh = (int)((4000.0/(FS/DOWNSAMPLE)*NFFT)+0.5);
//    int FreqbinPerThread = (int)((nfh-nfl+1)/(THREADNUMPERBLK*1.0) + 0.5);
    int BlockRowNum = 0;
    cufftComplex    *dev_fft=NULL;
    cufftReal       *dev_energy=NULL;
    cufftReal       *sum_energy=NULL;
    cufftComplex    *PhiArray = NULL;
    cufftHandle     Beamplan;
    float           c[BEAMNUM]={0.0};
    cufftComplex    *sk=NULL;
    float           *debugvar = NULL;
    int             peak[BEAMNUM]={0};
    int             valley[BEAMNUM]={0};
//    bool            traced[BEAMNUM] = {false};
//    int             tracedbeamIdx = -1;
    float           pretracedtarget[BEAMNUM] = {0.0};
    int             pretracedtargetIdx[BEAMNUM] = {-1};
    int             pretracedtargetNum = 0;
    int             tracedtargetbeam[MAXTRACETARNUM][2];
//    float           *tracebeam = NULL;
//    int             beammatrix[5][BEAMNUM] = {0};
    int             i0,i1,i2;
    float           r0,r1,r2;
    float           delta_index = 0;
    float           tracedtargetangle[3] = {0.0};
    cufftReal       *dev_delayFilter = NULL;
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
    //-------------------------------------------------------------
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
//    int retvalprint = 0;

    //------------------------------------------------------------
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
    //-----------------------------------------------------------------------------------------
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    while (1)
    {
        pthread_mutex_lock(&count_lock_FrameDataReady);
        while (count_FrameDataReady == 0)
        {
             pthread_cond_wait(&cond_FrameDataReady,&count_lock_FrameDataReady);
        }
        count_FrameDataReady = count_FrameDataReady -1;
        pthread_mutex_unlock(&count_lock_FrameDataReady);

        FrameNum++;

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

        //-----------------------------------------(1)信号滤波降采样---------------------------------------------------
        for(int jj=0;jj<CHANNUM*2;jj++)
        {
            cufftExecR2C(Xplan,(cufftReal *)&dev_x[jj*FILTER_FRAME],(cufftComplex *)&dev_fft_x[jj*FILTER_FRAME]);
        }

        DownSamplingFilter<<<CHANNUM*2*(FILTER_FRAME/2/THREADNUMPERBLK),THREADNUMPERBLK>>>(dev_fft_x,dev_fft_h,dev_fft_y,FILTER_FRAME);

        for(int jj=0;jj<CHANNUM*2;jj++)
        {
            cufftExecC2R(Yplan,(cufftComplex *)&dev_fft_y[jj*FILTER_FRAME],(cufftReal *)&dev_y[jj*FILTER_FRAME]);
            cudaMemcpy(dev_chanbuff+jj*FILTER_FRAME/DOWNSAMPLE,dev_chanbuff+jj*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2,FILTER_FRAME/DOWNSAMPLE/2*sizeof(float),cudaMemcpyDeviceToDevice);
        }
        IFFTNormalize<<<CHANNUM*2*(FILTER_FRAME/2/THREADNUMPERBLK),THREADNUMPERBLK>>>(dev_y,dev_chanbuff,FILTER_FRAME);
        //-----------------------------------------(1)信号滤波降采样结束---------------------------------------------------

        //-----------------------------------------(2)频域波束形成---------------------------------------------------
        for (int ii=0;ii<ARRAYNUM;ii++)
        {
            cufftExecR2C(Beamplan,(cufftReal *)&dev_chanbuff[ii*FILTER_FRAME/DOWNSAMPLE+FILTER_FRAME/DOWNSAMPLE/2],(cufftComplex *)&dev_fft[ii*(NFFT/2+1)]);
        }

        FD_Beamform<<<BlockRowNum*BEAMNUM,THREADNUMPERBLK>>>(dev_fft,dev_energy,PhiArray,nfl,nfh);
        MatrixSumRow<<<BEAMNUM,1>>>(dev_energy,sum_energy,BlockRowNum,BEAMNUM);
        printf("success 0!\n");
        cudaMemcpy(c,sum_energy,BEAMNUM*sizeof(float),cudaMemcpyDeviceToHost);
       // fwrite(c,sizeof(float),BEAMNUM,fp);
        //-----------------------------------------(2)频域波束形成结束-----------------------------------------------

        //-----------------------------------------(3)波束能量检测------------------------------------------
        //
        memset(peak,0,BEAMNUM*sizeof(int));
        memset(valley,0,BEAMNUM*sizeof(int));
        findpeak(c,peak,BEAMNUM);
        findvalley(c,valley,BEAMNUM);
        bool targetexist = false;

        memset(pretracedtarget,0,sizeof(float)*BEAMNUM);
        memset(pretracedtargetIdx,0,sizeof(int)*BEAMNUM);
        pretracedtargetNum = 0;

        for(int kk=0;kk<BEAMNUM;kk++)
        {
            if(peak[kk] == 1)
            {
                //
                int jj=0;
                for(jj=0;jj<MAXTRACETARNUM;jj++)
                {
                    //
                    if(abs(tracedtargetbeam[jj][0]-kk)<6 && tracedtargetbeam[jj][0]>0)
                    {
                        break;
                    }
                }
                if(jj==MAXTRACETARNUM)  //
                {
                    targetexist = peakdetection(kk,c,valley,2.0);
                }
                else  //
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
                //
                if(abs(tracedtargetbeam[jj][0]-pretracedtargetIdx[kk])<6 && tracedtargetbeam[jj][0]>0)
                {
                    tracedtargetbeam[jj][0] = pretracedtargetIdx[kk];
                    tracedtargetbeam[jj][1] = FrameNum;
                    break;
                }
            }

            if(jj==MAXTRACETARNUM)  //
            {
                int ii = 0;
                for(ii=0;ii<MAXTRACETARNUM;ii++)
                {
                    //
                    if(tracedtargetbeam[ii][0] < 0)
                    {
                        break;
                    }
                }
                if(ii < MAXTRACETARNUM)           //
                {
                    tracedtargetbeam[ii][0] = pretracedtargetIdx[kk];
                    tracedtargetbeam[ii][1] = FrameNum;
                }
            }
        }
        //
        for(int jj=0;jj<MAXTRACETARNUM;jj++)
        {
            if(tracedtargetbeam[jj][0] >0 && FrameNum - tracedtargetbeam[jj][1] >= 5)
            {
                tracedtargetbeam[jj][0] = -1;
                tracedtargetbeam[jj][1] = -1;
                tracedtargetangle[jj] = -1.0f;
            }
        }
        //-----------------------------------------(3)波束能量检测-------------------------------------

        //-----------------------------------------(4) 波束跟踪、跟踪波束 ------------------------------
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
                printf("success 1!\n");

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
                printf("success 2!\n");
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
                printf("success 3!\n");
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
        printf("success 4!\n");
        //-----------------------------------------(4) 波束跟踪、跟踪波束 ------------------------------------------
        //-----------------------------------------(5) 矢量处理----------------------------------------------------
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
        printf("success 5!\n");
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
        //------------------------------------------------------------------------------------------------
        cudaEventRecord(stop1,NULL);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&msecTotal,start1,stop1);

        printf("%d:%f;%d,%d;%d,%d;%d,%d\n",FrameNum,msecTotal,tracedtargetbeam[0][0],tracedtargetbeam[0][1],tracedtargetbeam[1][0],tracedtargetbeam[1][1],tracedtargetbeam[2][0],tracedtargetbeam[2][1]);
        printf("\n");
        fprintf(fplog,"%d:%f;%d,%d;%d,%d;%d,%d\n",FrameNum,msecTotal,tracedtargetbeam[0][0],tracedtargetbeam[0][1],tracedtargetbeam[1][0],tracedtargetbeam[1][1],tracedtargetbeam[2][0],tracedtargetbeam[2][1]);
        fflush(fplog);
        }
}

void *DataFormatting(void *lParam)
{
    //int retval1 = -1;
    //int retval2 = -1;
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

	while (1)
	{
//#if ONLINEMODE
//        pthread_mutex_lock(&count_lock_BoardDataReady);
//        while (count_BoardDataReady == 0)
//        {
//            pthread_cond_wait(&cond_BoardDataReady,&count_lock_BoardDataReady);
//        }
//        count_BoardDataReady = count_BoardDataReady -1;
//        pthread_mutex_unlock(&count_lock_BoardDataReady);
//#endif

//#if FILEMODE
        pthread_mutex_lock(&count_lock_Board1DataReady);
        while (count_Board1DataReady == 0)
        {
            pthread_cond_wait(&cond_Board1DataReady,&count_lock_Board1DataReady);
        }
        count_Board1DataReady = count_Board1DataReady -1;
        pthread_mutex_unlock(&count_lock_Board1DataReady);

        pthread_mutex_lock(&count_lock_Board2DataReady);
        while (count_Board2DataReady == 0)
        {
            pthread_cond_wait(&cond_Board2DataReady,&count_lock_Board2DataReady);
        }
        count_Board2DataReady = count_Board2DataReady -1;
        pthread_mutex_unlock(&count_lock_Board2DataReady);


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
            printf("DataFormatting Finished!\n");
            pthread_mutex_lock(&count_lock_FrameDataReady);
            pthread_cond_signal(&cond_FrameDataReady);
            count_FrameDataReady = count_FrameDataReady+1;
            pthread_mutex_unlock(&count_lock_FrameDataReady);
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
            printf("DataFormatting Finished!\n");
            pthread_mutex_lock(&count_lock_FrameDataReady);
            pthread_cond_signal(&cond_FrameDataReady);
            count_FrameDataReady = count_FrameDataReady+1;
            pthread_mutex_unlock(&count_lock_FrameDataReady);
        }
//#endif
	}

}

void *ReceiveNetwork(void *lParam)
{
	char errBuf[PCAP_ERRBUF_SIZE], *device;
	pcap_t *handle;
	bpf_u_int32 mask;
	bpf_u_int32 net;
	struct bpf_program filter;
	char filter_app[] = "udp dst port 0"; //setting the filter package
    struct pcap_pkthdr packet;
    const u_char *pktStr;
    char packtype = 0;
	short portnumber = 0;
    char sourceid = 0;
    char FramenumN1 = -1, FramenumN2 = -1;
    char LastFramenumN1 = 0, LastFramenumN2 = 0;
	int readbufb1[TL*CHANNUM+1],readbufb2[TL*CHANNUM+1];
    int BUF_FLAG_B1=0,BUF_FLAG_B2;
    int *pBuf_B1 = NULL,*pBuf_B2 = NULL;
    int *pCounter_B1 = NULL,*pCounter_B2 = NULL;
    int CounterA_B1 = FRAMELEN,CounterB_B1 = FRAMELEN;
    int CounterA_B2 = FRAMELEN,CounterB_B2 = FRAMELEN;
	int temp = 0;
	int FrameNum1 = 0,FrameNum2 = 0, FrameNum = 0;

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


	//get the name of the first device suitable for capture
	device = pcap_lookupdev(errBuf);
	if ( device )
	{
		printf("success: device: %s\n",device);
	}
	else
	{
		printf("error: %s\n",errBuf);
		return 0;
	}

	//open network device for packet capture
	handle = pcap_open_live(device,BUFSIZ,1,0,errBuf);

	//look up into from the capture device
	pcap_lookupnet(device,&net,&mask,errBuf);
	printf("net=%x mask=%x\n",net,mask);	

    //compiles the filter expression into a bpf filter rogram
    printf("compiles the filter expression into a bpf filter program\r\n");
    pcap_compile(handle,&filter,filter_app,0,net);
    
    //load the filter program into the packet capture device
    printf("load the filter program into the packet capture device\r\n");
    pcap_setfilter(handle,&filter);



	while (1)
	{
        //printf("before Received data!\n");
		pktStr = pcap_next(handle,&packet);
        //printf("Received data!\n");

		if(pktStr != NULL)
		{
            //printf("Received data!\n");
            //读取目的端口号
            memcpy((char *)&portnumber,pktStr+37,sizeof(char));
            memcpy((char *)&portnumber+1,pktStr+36,sizeof(char));
			if (portnumber == DEST_PORT)
			{
				//读取包类型
                memcpy(&packtype,pktStr+45,sizeof(char));
                memcpy(&sourceid,pktStr+43,sizeof(char));
				if (packtype == 0x10)  // if packet is ADC packet
				{
					if(sourceid == 1)
					{
						FrameNum1++;
                        memcpy(readbufb1,pktStr+42,(TL*CHANNUM+1)*sizeof(int));
                        FramenumN1 = *(pktStr+44);
						FramenumN1 = FramenumN1 >> 2;
						if (FrameNum1 == 1)
						{
							LastFramenumN1 = FramenumN1;
						}
						else
						{
							if (FramenumN1 != LastFramenumN1+1 && FramenumN1+63 != LastFramenumN1)
							{							
								printf("Lost Board1 data package!\n");
							}
							LastFramenumN1 = FramenumN1;
						}
					}
					if(sourceid == 2)
					{
						FrameNum2++;
                        memcpy(readbufb2,pktStr+42,(TL*CHANNUM+1)*sizeof(int));
                        FramenumN2 = *(pktStr+44);
						FramenumN2 = FramenumN2 >> 2;
						if (FrameNum2 == 1)
						{
							LastFramenumN2 = FramenumN2;
						}
						else
						{
							if (FramenumN2 != LastFramenumN2+1 && FramenumN2+63 != LastFramenumN2)
							{
								printf("Lost Board2 data package!\n");
							}
							LastFramenumN2 = FramenumN2;
						}
					}
					if (FramenumN1 == FramenumN2 && FramenumN2 >= 0)  //receive both board data
					{
						//-----------------board1 data accumulate---------------------------
						if(0 == BUF_FLAG_B1)
						{
							pBuf_B1 = DataBufA_B1;
							pCounter_B1 = &CounterA_B1;
						}
						else
						{
							pBuf_B1 = DataBufB_B1;
							pCounter_B1 = &CounterB_B1;
						}
						if(*(pCounter_B1)>=TL) //
						{
							memcpy(pBuf_B1+FRAMELEN*CHANNUM-(*(pCounter_B1))*CHANNUM,readbufb1+1,TL*CHANNUM*sizeof(int));
							*(pCounter_B1) = *(pCounter_B1)-TL;
						}
						else
						{
							temp = TL - *(pCounter_B1);
							memcpy(pBuf_B1+FRAMELEN*CHANNUM-(*(pCounter_B1))*CHANNUM,readbufb1+1,(*(pCounter_B1))*CHANNUM*sizeof(int));
							*(pCounter_B1)= FRAMELEN;
							if(0 == BUF_FLAG_B1)
							{
								memcpy(DataBufB_B1+FRAMELEN*CHANNUM-CounterB_B1*CHANNUM,readbufb1+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
								CounterB_B1 = CounterB_B1 - temp;
								BUF_FLAG_B1 = 1;
							}
							else //
							{
								memcpy(DataBufA_B1+FRAMELEN*CHANNUM-CounterA_B1*CHANNUM,readbufb1+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
								CounterA_B1 = CounterA_B1 - temp;
								BUF_FLAG_B1 = 0;
							}
							pthread_mutex_lock(&count_lock_Board1DataReady);
							pthread_cond_signal(&cond_Board1DataReady);
							count_Board1DataReady = count_Board1DataReady+1;
							pthread_mutex_unlock(&count_lock_Board1DataReady);
//                                                        printf("ReceiveNetworkData A Finished!\n");
						}
						//-----------------board2 data accumulate---------------------------
						if(0 == BUF_FLAG_B2)
						{
							pBuf_B2 = DataBufA_B2;
							pCounter_B2 = &CounterA_B2;
						}
						else
						{
							pBuf_B2 = DataBufB_B2;
							pCounter_B2 = &CounterB_B2;
						}
						if(*(pCounter_B2)>=TL) //
						{
							memcpy(pBuf_B2+FRAMELEN*CHANNUM-(*(pCounter_B2))*CHANNUM,readbufb2+1,TL*CHANNUM*sizeof(int));
							*(pCounter_B2) = *(pCounter_B2)-TL;
						}
						else
						{
							temp = TL - *(pCounter_B2);
							memcpy(pBuf_B2+FRAMELEN*CHANNUM-(*(pCounter_B2))*CHANNUM,readbufb2+1,(*(pCounter_B2))*CHANNUM*sizeof(int));
							*(pCounter_B2)= FRAMELEN;
							if(0 == BUF_FLAG_B2)
							{
								memcpy(DataBufB_B2+FRAMELEN*CHANNUM-CounterB_B2*CHANNUM,readbufb2+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
								CounterB_B2 = CounterB_B2 - temp;
								BUF_FLAG_B2 = 1;
							}
							else 
							{
								memcpy(DataBufA_B2+FRAMELEN*CHANNUM-CounterA_B2*CHANNUM,readbufb2+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
								CounterA_B2 = CounterA_B2 - temp;
								BUF_FLAG_B2 = 0;
							}
							pthread_mutex_lock(&count_lock_Board2DataReady);
							pthread_cond_signal(&cond_Board2DataReady);
							count_Board2DataReady = count_Board2DataReady+1;
							pthread_mutex_unlock(&count_lock_Board2DataReady);
//                                                        printf("ReceiveNetworkData B Finished!\n");
						}
					}
				}
			}
		}
		//printf("ReceiveNetworkData Finished!\n");
		//pthread_mutex_lock(&count_lock_BoardDataReady);
		//pthread_cond_signal(&cond_BoardDataReady);
		//count_BoardDataReady = count_BoardDataReady+1;
		//pthread_mutex_unlock(&count_lock_BoardDataReady);
	}
}

void *ReadBoard1Data(void *lParam)
{
    int fileindex = 0;
    std::string FilePath = "/home/ubuntu/Desktop/GPU/uwrn/";
    std::string FileNamePre = "Board1_ADC_";
    std::string FileIdx = std::to_string(fileindex);
    std::string FileNameSur = ".bin";
    std::string FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
    int DataFileNum = 18;
    FILE *fp = NULL;
    //int readbytes = 0;
    int readbuf[TL*CHANNUM+1];
    int BUF_FLAG=0;
    int *pBuf = NULL;
    int *pCounter = NULL;
    int CounterA = FRAMELEN,CounterB = FRAMELEN;
    int temp = 0;


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

    //
    for(int ii=0;ii<DataFileNum;ii++)
    {
        fileindex = ii;
        FileIdx = std::to_string(fileindex);
        FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
        if(fp != NULL)
        {
            fclose(fp);
            fp = NULL;
        }
        fp = fopen(FileName.c_str(),"rb");
        for(int jj=0;jj<8e4;jj++)
        {
            usleep(TL*1e6 / FS);
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
            if(*(pCounter)>=TL) //
            {
                memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,TL*CHANNUM*sizeof(int));
                *(pCounter) = *(pCounter)-TL;
            }
            else
            {
                temp = TL - *(pCounter);
                //
                memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,(*(pCounter))*CHANNUM*sizeof(int));
                //
                *(pCounter)= FRAMELEN;
                //
                if(0 == BUF_FLAG) //
                {
                    memcpy(DataBufB_B1+FRAMELEN*CHANNUM-CounterB*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
                    //
                    CounterB = CounterB - temp;
                    //
                    BUF_FLAG = 1;
                }
                else //
                {
                    memcpy(DataBufA_B1+FRAMELEN*CHANNUM-CounterA*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
                    //
                    CounterA = CounterA - temp;
                    //
                    BUF_FLAG = 0;
                }
                //
                //SetEvent(g_hReadBoard1ThreadReadyEnvent);
                pthread_mutex_lock(&count_lock_Board1DataReady);
                pthread_cond_signal(&cond_Board1DataReady);
                count_Board1DataReady = count_Board1DataReady+1;
                pthread_mutex_unlock(&count_lock_Board1DataReady);
            }
        }
    }
    return NULL;
}

void *ReadBoard2Data(void *lParam)
{
    int fileindex = 0;
    std::string FilePath = "/home/ubuntu/Desktop/GPU/uwrn/";
    std::string FileNamePre = "Board2_ADC_";
    std::string FileIdx = std::to_string(fileindex);
    std::string FileNameSur = ".bin";
    std::string FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
    int DataFileNum = 18;
    FILE *fp = NULL;
    //int readbytes = 0;
    int readbuf[TL*CHANNUM+1];
    int BUF_FLAG=0;
    int *pBuf = NULL;
    int *pCounter = NULL;
    int CounterA = FRAMELEN,CounterB = FRAMELEN;
    int temp = 0;

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

    //
    for(int ii=0;ii<DataFileNum;ii++)
    {
        fileindex = ii;
        FileIdx = std::to_string(fileindex);
        FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
        if(fp != NULL)
        {
            fclose(fp);
            fp = NULL;
        }
        fp = fopen(FileName.c_str(),"rb");
        for(int jj=0;jj<8e4;jj++)
        {
            usleep(TL*1e6 / FS);
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
            if(*(pCounter)>=TL) //
            {
                memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,TL*CHANNUM*sizeof(int));
                *(pCounter) = *(pCounter)-TL;
            }
            else
            {
                temp = TL - *(pCounter);
                //
                memcpy(pBuf+FRAMELEN*CHANNUM-(*(pCounter))*CHANNUM,readbuf+1,(*(pCounter))*CHANNUM*sizeof(int));
                //
                *(pCounter)= FRAMELEN;
                //
                if(0 == BUF_FLAG) //
                {
                    memcpy(DataBufB_B2+FRAMELEN*CHANNUM-CounterB*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
                    //
                    CounterB = CounterB - temp;
                    //
                    BUF_FLAG = 1;
                }
                else //
                {
                    memcpy(DataBufA_B2+FRAMELEN*CHANNUM-CounterA*CHANNUM,readbuf+(TL-temp)*CHANNUM+1,temp*CHANNUM*sizeof(int));
                    //
                    CounterA = CounterA - temp;
                    //
                    BUF_FLAG = 0;
                }
                //
                //SetEvent(g_hReadBoard2ThreadReadyEnvent);
                pthread_mutex_lock(&count_lock_Board2DataReady);
                pthread_cond_signal(&cond_Board2DataReady);
                count_Board2DataReady = count_Board2DataReady+1;
                pthread_mutex_unlock(&count_lock_Board2DataReady);
            }
        }
    }
    return NULL;
}


