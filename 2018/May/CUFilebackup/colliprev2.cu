//nvcc colliprev.cu -o test -lstdc++ -lpthread -lcufft -lpcap -std=c++11
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
#include <cufft.h>
//-------------------------------------

// ----------------------------------------
#define     PI			     3.1415926f
#define     UWC			     1500.0f          //
#define     FS			     250000           //
#define     threadsPerBlock          512
#define     d                        0.07f
#define     FL                       90000.0f
#define     FH                       100000.0f
#define     TL                       17
#define     CHANNUM                  16
#define     FRAMELEN                 6800
#define     DOWNSAMPLE               1
#define     FIRORDER                 256
#define     FILTER_FRAME             FRAMELEN
#define     NFFT		     FRAMELEN	      //
#define     BEAMNUM                  91
#define     THREADNUMPERBLK          200
#define     ARRAYNUM                 15
#define     STARTBEAM                15
#define     ENDBEAM                  75
#define     MAXTRACETARNUM           3
#define     M                        3
#define     ONLINEMODE               0
#define     FILEMODE                 1
#define     DEST_PORT                0
#define     PSD_LEN                  20

#define	    THETANUM		     3                       //俯仰角个数
#define	    PHINUM		     9                       //方位角个数
#define	    NREF		     250                     //参考信号长度
#define     FRAMELEN                 6800
#define	    NHILBT		     FRAMELEN                //希尔伯特变换频点数
#define	    ELENUM   		     12                      //阵元个数
#define	    NNZERO    		     200	             //信号补零点数（弥补滤波器时延）
#define	    NFIR		     (FILTER_FRAME+NNZERO)    //滤波点数
#define	    NMAT   		     FRAMELEN		     //匹配滤波器点数
#define     DIRECTARRIVENUM          30
// -----------------------------------------------------
void *ReadBoard0Data(void *lParam);
void *ActiveReceiveNetwork(void *lParam);
void *ActiveDataFormatting(void *lParam);
void *ActiveSignalProcessing(void *lParam);
//------------------------------------------------------

pthread_mutex_t count_lock_Board0DataReady;
pthread_mutex_t count_lock_ActiveFrameDataReady;

pthread_cond_t  cond_Board0DataReady;
pthread_cond_t  cond_ActiveFrameDataReady;

unsigned int    count_Board0DataReady;
unsigned int    count_ActiveFrameDataReady;
//-----------------------------------------------------
int *DataBufA_B1 = NULL;//16Channel
float *ChannDataBufA=NULL;//16Channel
float *ChannDataBuf=NULL;//12Channel

//---------------------------------------------------
int   fir1(int n,int band,float fl,float fh,float fs,int wn, float *h);
float window(int type,int n,int i,float beta);
float kaiser(int i,int n,float beta);
float bessel0(float x);
void  findpeak(float *data, int *p,int dn);
void  findvalley(float *data, int *p,int dn);
bool  peakdetection(int beamidx,float *be,int *valley,float threshold);
void  rbub(float *p,int *idx,int n);
// -----------------------------------------------------------
float rsRef[NREF]={0.0};//翻转参考信号
float theta[3]={1.3963, 1.5708, 1.7453};//俯仰角:80°-100°
float phi[9]={0.8727, 1.0472, 1.2217, 1.3963 ,  1.5708,  1.7453 ,  1.9199 , 2.0944 , 2.2689};//方位角
float xEle[12]={0.0};//阵元x坐标
float zEle[12]={0.0};//阵元z坐标
float dTime[THETANUM*PHINUM*ELENUM]={0.0};//延时
// -----------------------------------------------------------
int main(){
	pthread_t t_ActiveReceiveNetworkData;
        pthread_t t_ActiveDataFormatting;
        pthread_t t_ActiveSignalProcessing;
        pthread_t t_ReadBoard0Data;

	cond_Board0DataReady      = PTHREAD_COND_INITIALIZER;
        cond_ActiveFrameDataReady = PTHREAD_COND_INITIALIZER;

	count_lock_Board0DataReady = PTHREAD_MUTEX_INITIALIZER;
        count_lock_ActiveFrameDataReady = PTHREAD_MUTEX_INITIALIZER;

	pthread_create(&t_ActiveSignalProcessing,NULL,ActiveSignalProcessing,(void *)NULL);
        pthread_create(&t_ActiveDataFormatting,NULL,ActiveDataFormatting,(void *)NULL);

#if ONLINEMODE
    pthread_create(&t_ActiveReceiveNetworkData,NULL,ActiveReceiveNetwork,(void *)NULL);
#endif

#if FILEMODE
    pthread_create(&t_ReadBoard0Data,NULL,ReadBoard0Data,(void *)NULL);
#endif
	pthread_join(t_ActiveSignalProcessing, NULL);
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
__global__ void DownSamplingFilter(cufftComplex *dev_fft_sig,cufftComplex *dev_fft_filter,cufftComplex *dev_fft_yk,int FFTN)//needchange
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
__global__ void ActiveFilter(cufftComplex *dev_fft_sig,cufftComplex *dev_fft_filter,cufftComplex *dev_fft_yk,int FFTN)
{
        int bid = 0,tid = 0;
        cuComplex Sigk;
        cuComplex Hk;
        int chanIdx = 0;
        int freqIdx = 0;

        bid = blockIdx.x;
        tid = threadIdx.x;
        chanIdx = bid % (ELENUM);
        freqIdx = bid / ELENUM*THREADNUMPERBLK+tid;

        Sigk.x = dev_fft_sig[chanIdx*FFTN+freqIdx].x;
        Sigk.y = dev_fft_sig[chanIdx*FFTN+freqIdx].y;
        Hk.x = dev_fft_filter[freqIdx].x;
        Hk.y = dev_fft_filter[freqIdx].y;
        dev_fft_yk[chanIdx*FFTN+freqIdx].x = Sigk.x*Hk.x-Sigk.y*Hk.y;
        dev_fft_yk[chanIdx*FFTN+freqIdx].y = Sigk.x*Hk.y+Sigk.y*Hk.x;
}
__global__ void ActiveDelayFilterGen(cufftReal *h, int *dI, cufftReal *dF, float *delaytime,int index)//所有通道小数延时滤波器参数
{
        // h-滤波器参数
        // dI-整数延时
        // dF-小数延时
        // index-波束标号
        int bid = 0,tid = 0;
        int k=0;
        float dfs = 0.0;
        int DI = 0;
        __shared__ float sum;
        //__shared__ float dfs;

        bid = blockIdx.x;//通道标号
        tid = threadIdx.x;//滤波器系数标号

        if(tid == 0)
        {
                sum = 0.0;
                dfs = delaytime[index+bid];//延时
                DI = int(dfs);//整数延时
                dF[bid] =dfs-DI;//小数延时
                dI[bid] = DI;
                //printf("bid=%d,m=%d,theta = %.3f,dfs = %.3f,DI = %d\n",bid,m,theta,dfs,DI);
        }

        //块内线程同步
        __syncthreads();

        k = tid-M;
        h[bid*(2*M+1)+tid] = sin(k*1.0*PI-dF[bid]*PI+0.000001)/(k*1.0*PI-dF[bid]*PI+0.000001);

        //块内线程同步
        __syncthreads();

        if(tid == 0)
        {
                for(int k=0;k<2*M+1;k++)
                {
                        sum = sum + h[bid*(2*M+1)+k];
                }
        }
        __syncthreads();

        h[bid*(2*M+1)+tid] =  h[bid*(2*M+1)+tid]/sum;
}

__global__ void ActiveFineDelayFilter(cufftReal *dev_xin,cufftReal *dev_yout,cufftReal *delayfilter)//小数时延滤波器
{
        int bid,tid;
        float x=0.0,h=0.0;
        float sum = 0.0;

        bid = blockIdx.x;//数据标号
        tid = threadIdx.x;//滤波器系数标号
        __shared__ float y[2*M+1];

        if(tid == 0)
        {
                for(int ii=0;ii<2*M+1;ii++)
                {
                        y[ii] = 0.0;
                }
        }

        if(bid+tid >= M && bid+tid <= FRAMELEN+M)
        {
                x = dev_xin[bid-M+tid];
        }
        if(2*M-tid >=0)
        {
                h = delayfilter[2*M-tid];
        }
        y[tid] = x*h;

        //块内线程同步
        __syncthreads();
        if(tid == 0)
        {
                sum = 0.0;
                for(int jj=0;jj<2*M+1;jj++)
                {
                        sum = sum + y[jj];
                }
                dev_yout[bid] = sum;
        }
}

__global__ void VectorMultiplier(cufftComplex *dev_in,cufftComplex *dev_h,cufftComplex *dev_out)
{
        int bid=blockIdx.x;
        dev_out[bid].x=dev_in[bid].x*dev_h[bid].x-dev_in[bid].y*dev_h[bid].y;
        dev_out[bid].y=dev_in[bid].x*dev_h[bid].y+dev_in[bid].y*dev_h[bid].x;
}

__global__ void HilbFilt(cufftComplex *dev_hilboutfreq, cufftComplex *dev_matchdatafreq, int mid) //Hilbert频域变换
{
        int bid=blockIdx.x;
        float xx=dev_matchdatafreq[bid].x;
        float yy=dev_matchdatafreq[bid].y;
        if(bid<=mid)
        {
                dev_hilboutfreq[bid].x=yy;
                dev_hilboutfreq[bid].y=-xx;
        }
        else{
                dev_hilboutfreq[bid].x=-yy;
                dev_hilboutfreq[bid].y=xx;
        }
}
__global__ void DevFindPeak(cufftReal *dev_beamdata,int *dev_peak,int datalen)
{
        int acc=0,acc1=0;
        int i,j;
        float a0=0.0,a1=0.0;
        int bid=blockIdx.x;
//	int tid=threadIdx.x;

        for(i=0;i<datalen;i++)
        {
                a0=*(dev_beamdata+bid*datalen+i);
                //先向前找
                for(j=1;j<11;j++)
                {
                        if ((i+j)>=datalen)
                        {
                                a1=*(dev_beamdata+bid*datalen+i+j-datalen);
                        }
                        else
                        {
                                a1=*(dev_beamdata+bid*datalen+i+j);
                        }
                        if (a0>a1)
                        {
                                acc=acc+1;
                        }
                }
                a0=*(dev_beamdata+bid*datalen+i);
                ////再向后找
                for(j=1;j<11;j++)
                {
                        if ((i-j)<0)
                        {
                                a1=*(dev_beamdata+bid*datalen+i-j+datalen);
                        }
                        else
                        {
                                a1=*(dev_beamdata+bid*datalen+i-j);
                        }
                        if (a0>a1)
                        {
                                acc1=acc1+1;
                        }
                }
                if ((acc==10) && (acc1==10))
                {
                  //if(bid == 0)
                  //{
                         // printf("%d:%.1f\n",i,*(dev_beamdata+bid*datalen+i));
                  //}
                    *(dev_peak+bid*datalen+i)=1;
                }
                acc=0;
                acc1=0;
        }
}

__global__ void DevFindValley(cufftReal *dev_beamdata,int *dev_valley,int datalen)
{
        int acc=0,acc1=0;
        int i,j;
        float a0=0.0,a1=0.0;
        int bid=blockIdx.x;

        for(i=0;i<datalen;i++)
        {
                a0=*(dev_beamdata+bid*datalen+i);
                //先向前找
        for(j=1;j<6;j++)
                {
                        if ((i+j)>=datalen)
                        {
                                //a1=*(data+i+j-dn);
                                break;
                        }
                        else
                        {
                                a1=*(dev_beamdata+bid*datalen+i+j);
                        }
                        if (a0<a1)
                        {
                                acc=acc+1;
                        }
                }
                if(j<5)  //循环因break退出
                {
                        acc = 5;
                }
        a0=*(dev_beamdata+bid*datalen+i);
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
                                a1=*(dev_beamdata+bid*datalen+i-j);
                        }
                        if (a0<a1)
                        {
                                acc1=acc1+1;
                        }
                }
                if(j<5)  //循环因break退出
                {
                        acc1 = 5;
                }
                if ((acc==5) && (acc1==5))
                {
                    *(dev_valley+bid*datalen+i)=1;
                }
                acc=0;
                acc1=0;
        }
}

__global__ void DevPeakDetection(int *dev_peak,int *dev_valley,cufftReal *dev_beamdata,cufftReal *dev_preselected,cufftReal *dev_selected,int datalen,float threshold,float thresholdabs)
{
        int bid = blockIdx.x;
        int tid = threadIdx.x;
        int index = 0,ll=0;
        float pvr1 = 1.0,pvr2 = 1.0;
        bool foundfirst = false;
        float maxval = 0.0,c=1500.0;

        for(int ii=1;ii<datalen-1;ii++)
        {
                if(dev_peak[bid*datalen+ii] ==1)
                {
                        for(ll=ii+1;ll<datalen;ll++)
                        {
                                if(dev_valley[bid*datalen+ll] == 1)
                                {
                                        index = ll;
                                        break;
                                }
                        }
                        if(ll<=datalen-1)
                        {
                                pvr1 = dev_beamdata[bid*datalen+ii] / dev_beamdata[bid*datalen+index];
                        }

                        for(ll=ii-1;ll>=0;ll--)
                        {
                                if(dev_valley[bid*datalen+ll] == 1)
                                {
                                        index = ll;
                                        break;
                                }
                        }
                        if(ll>=0)
                        {
                                pvr2 = dev_beamdata[bid*datalen+ii] / dev_beamdata[bid*datalen+index];
                        }

                        if(pvr1 >= threshold && pvr2 >= threshold && dev_beamdata[bid*datalen+ii] > thresholdabs)
                        {
                                dev_preselected[bid*datalen+ii]=1;
                        }
                        else
                        {
                                dev_preselected[bid*datalen+ii]=0;
                        }
                }
                else
                {
                        dev_preselected[bid*datalen+ii]=0;
                }
        }
        //找第一个峰值和最大峰值
        for(int ii=0;ii<datalen-1;ii++)
        {
                if(dev_preselected[bid*datalen+ii] == 1 && foundfirst == false)
                {
                        foundfirst = true;
                        dev_selected[bid*4+0] = (DIRECTARRIVENUM*TL+ii) *1.0 / FS * c / 2;
                        //dev_selected[bid*3+0] = ii;
                        dev_selected[bid*4+1] = dev_beamdata[bid*datalen+ii];
                }
                if(dev_beamdata[bid*datalen+ii] > maxval)
                {
                        dev_selected[bid*4+2] = (DIRECTARRIVENUM*TL+ii) *1.0 / FS * c / 2;
                        //dev_selected[bid*3+2] = ii;
                        dev_selected[bid*4+3] = dev_beamdata[bid*datalen+ii];
                        maxval = dev_beamdata[bid*datalen+ii];
                }
        }
      //  if(bid == 4)
      //  {
        	//printf("%d:%.1f,%.1f,%.1f,%.1f\n",bid,dev_selected[bid*4+0],dev_selected[bid*4+1],dev_selected[bid*4+2],dev_selected[bid*4+3]);
	//	printf("#%d: %.1f\n",bid,dev_selected[bid*4+2]);
      //  }
}
__global__ void Envelope(cufftReal *dev_envelopedata, cufftReal *dev_delayfilterout, cufftReal *dev_hilbout) //求包络
{
        int bid=blockIdx.x;
        float xx=dev_delayfilterout[bid];
        float yy=dev_hilbout[bid]/FRAMELEN;//一定要归一化！！！！
        dev_envelopedata[bid]=sqrt(xx*xx+yy*yy);
}

void *ReadBoard0Data(void *lParam){
	int fileindex = 0;
        std::string FilePath = "/home/ubuntu/Documents/Active/";     //数据文件路径，根据需要更改
        std::string FileNamePre = "Board0_ADC_";
        std::string FileIdx = std::to_string(fileindex);
        std::string FileNameSur = ".bin";
        std::string FileName = FilePath + FileNamePre + FileIdx + FileNameSur;
        int DataFileNum = 1;
        FILE *fp = NULL;

        int readbuf[TL*CHANNUM+3];
        int CounterA = FRAMELEN;
        int temp = 0;
        bool foundpulse = false;

        //QueryPerformanceFrequency(&nFreq);

        if(DataBufA_B1 != NULL)
        {
                free(DataBufA_B1);
                DataBufA_B1 = NULL;
        }
        DataBufA_B1 = (int *)malloc(FRAMELEN*CHANNUM*sizeof(int));
        memset(DataBufA_B1,0,FRAMELEN*CHANNUM*sizeof(int));

	
        //QueryPerformanceCounter(&nBeginTime);
        //每次读取1个数据包，即17samples*16channels，数据类型为24bit整型，以int型存储
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
                int num=0;
                for(int jj=0;jj<8e4;jj++)//
                {
                        usleep(TL*1e6 / FS);//wait
                        fread(readbuf,sizeof(int),TL*CHANNUM+3,fp);
                        //搜索脉冲前沿
                        if(!foundpulse)
                        {                              
				 float dataval = 0.0;
                                for(int kk=0;kk<TL;kk++)
                                {
                                        temp = readbuf[3+kk*CHANNUM+1];//2通道
                                        temp = temp<<8;
                                        temp = temp>>8;
                                        dataval = temp*1.0/pow(2.0,23) * 2.5;
                                        if(fabs(dataval) > 0.5)
                                        {
                                                foundpulse = true;
                                                break;
                                        }
                                }
                        }
                        if(foundpulse && num++ > DIRECTARRIVENUM) //去掉直达波后510个点  510/17=30
                        {
                                memcpy(DataBufA_B1+FRAMELEN*CHANNUM-CounterA*CHANNUM,readbuf+3,TL*CHANNUM*sizeof(int));
                                CounterA = CounterA-TL;
                                if(CounterA == 0)
                                {
                                        //使事件有效
                                        pthread_mutex_lock(&count_lock_Board0DataReady);
                                        pthread_cond_signal(&cond_Board0DataReady);
                                        count_Board0DataReady = count_Board0DataReady+1;
                                        pthread_mutex_unlock(&count_lock_Board0DataReady);
                                        foundpulse = false;
                                        CounterA = FRAMELEN;
                                        num=0;
					//printf("readboard0data.\n");
                                }
                        }
                }
        }
	return NULL;
}

void *ActiveDataFormatting(void *lParam)
{
        int temp = 0;
//	int index=0;
//	FILE *fp=NULL;
	

        if(ChannDataBufA != NULL)
        {
                free(ChannDataBufA);
                ChannDataBufA = NULL;
        }
        ChannDataBufA = (float *)malloc(FRAMELEN*CHANNUM*sizeof(float));
        memset(ChannDataBufA,0,FRAMELEN*CHANNUM*sizeof(float));

        if(ChannDataBuf != NULL)
        {
                free(ChannDataBuf);
                ChannDataBuf = NULL;
        }
        ChannDataBuf = (float *)malloc(FRAMELEN*ELENUM*sizeof(float));
        memset(ChannDataBuf,0,FRAMELEN*ELENUM*sizeof(float));

        while(1)
        {
                pthread_mutex_lock(&count_lock_Board0DataReady);
                while (count_Board0DataReady == 0)
                {
                    pthread_cond_wait(&cond_Board0DataReady,&count_lock_Board0DataReady);
                } 
                count_Board0DataReady = count_Board0DataReady -1;
                pthread_mutex_unlock(&count_lock_Board0DataReady);
                for(int ii=0;ii<CHANNUM;ii++)
                {
                        for(int jj=0;jj<FRAMELEN;jj++)
                        {
                                temp = DataBufA_B1[jj*CHANNUM+ii];
                                temp = temp<<8;
                                temp = temp>>8;
                                if(ii==1 || ii==6 || ii==7 || ii==9 || ii==11)
                                {
                                        ChannDataBufA[ii*FRAMELEN+jj] = -temp*1.0/pow(2.0,23) * 2.5;
                                }
                                else
                                {
                                        ChannDataBufA[ii*FRAMELEN+jj] = temp*1.0/pow(2.0,23) * 2.5;
                                }
                        }
                }
		//去掉多余4个通道的数据
                memcpy(ChannDataBuf,ChannDataBufA,sizeof(float)*ELENUM*FRAMELEN);

                pthread_mutex_lock(&count_lock_ActiveFrameDataReady);
                pthread_cond_signal(&cond_ActiveFrameDataReady);
                count_ActiveFrameDataReady = count_ActiveFrameDataReady+1;
                pthread_mutex_unlock(&count_lock_ActiveFrameDataReady);
		

//	std::string fname="/home/ubuntu/Documents/Active/tmp/formdata"+std::to_string(index++)+".bin";
//	fp=fopen(fname.c_str(),"wb");
//	fwrite(ChannDataBuf,sizeof(float),12*FRAMELEN,fp);
//	fclose(fp);
//	fp=NULL;
        }
}
void InitProcessing(){

//-----------------------------------Init();-------------------------------
        //翻转参考信号：频域匹配
        for(int ii=0;ii<NREF;ii++)
        {
                float t=1.0*ii/FS;
                rsRef[NREF-1-ii]=sin(2*PI*(90e3*t+0.5e7*t*t));
        }
        //阵元坐标
        for(int jj=0;jj<6;jj++)
        {
                xEle[jj]=23e-3*sin(jj*PI/3);
                zEle[jj]=23e-3*cos(jj*PI/3);
        }
        for(int jj=0;jj<6;jj++)
        {
                xEle[6+jj]=11.5e-3*sin(jj*PI/3);
                zEle[6+jj]=11.5e-3*cos(jj*PI/3);
        }
        //延时
        for(int ii=0;ii<3;ii++)
        {
                for(int jj=0;jj<9;jj++)
                {
                        for (int kk=0;kk<12;kk++)
                        {
                                dTime[(ii*9+jj)*12+kk] = (xEle[kk]*sin(theta[ii])*cos(phi[jj])+zEle[kk]*cos(theta[ii]))/UWC*FS;
                        }
                }
        }
//---------------------------------------------------Init finished-----------------------------


}

void *ActiveSignalProcessing(void *lParam)
{
	float temp[27*FRAMELEN]={0.0};

	InitProcessing();
        int FrameNum = 0;
        int fIndex=0;
        FILE * wfp=NULL;
        std::string FileName="";
        //-----------------滤波参数-------------------------------
        float h[FIRORDER+1] = {0.0};
        float fl = 80e3f,fh = 120e3f;
        cudaError    cudaStatus;
        cufftReal    *dev_x=NULL;              //12通道原始数据
        cufftReal	 *dev_x_s=NULL;				 //单个通道原始数据：后面需要补零
        cufftReal    *dev_h=NULL;              //滤波器系数
        cufftComplex *dev_fft_x=NULL;          //12通道原始数据FFT
        cufftComplex *dev_fft_h=NULL;          //滤波器系数FFT
        cufftComplex *dev_fft_y=NULL;          //滤波器输出FFT
        cufftReal    *dev_y=NULL;              //滤波器输出原始采样率时域信号
        cufftReal    *dev_chanbuff=NULL;       //显存内数据缓冲区
        //float        *FilteredDataout = NULL;
       // float        *DownSamplingData = NULL;
        cufftHandle  Hplan;                    //滤波器系数FFT
        cufftHandle  Xplan;                    //通道原始数据FFT
        cufftHandle  Yplan;                    //滤波后通道数据FFT
        cufftHandle	 HXplan;				   //Hilbert原始数据FFT
        cufftHandle	 HYplan;				   //Hilbert滤波后数据FFT
        cufftHandle  MXplan;
        cufftHandle	 MHplan;				   //匹配滤波器系数
        cufftHandle  MYplan;
        //_Longlong FiltDataFileIndex=0;
        //----------------------------------------------------------------
        //-----------------波束形成参数-------------------------------
        cufftReal *dev_dTime=NULL;
        cufftReal *dev_mat=NULL;//匹配滤波器系数
        cufftComplex *dev_fft_mat=NULL;//匹配滤波器频响
        float *beamdata=NULL;
        beamdata= (float *)malloc(FRAMELEN*PHINUM*THETANUM*sizeof(float));
        memset(beamdata,0,FRAMELEN*PHINUM*THETANUM*sizeof(float));
	float		 *selected=NULL;
	selected=(float *)malloc(4*PHINUM*THETANUM*sizeof(float));
	memset(selected,0,4*PHINUM*THETANUM*sizeof(float));

        cufftReal	 *dev_delayfilterbuf=NULL;				 //所有通道整数延时数据
        cufftReal	 *dev_delayfilterout=NULL;				 //所有通道精细延时数据
        cufftReal        *dev_delayFilter=NULL;		//延时滤波器参数
        int	         *dev_dI=NULL;						//所有通道整数时延
        cufftReal	 *dev_dF=NULL;				//所有通道小数时延
        cufftReal	 *dev_delaydata=NULL;	    //波束形成结果

        cufftReal	 *dev_matchdata=NULL;		//匹配滤波结果
        cufftReal        *dev_beamdata=NULL;
        cufftComplex     *dev_fft_delaydata=NULL;	//匹配滤波器输入
        cufftComplex     *dev_fft_matout=NULL;		//匹配滤波器输出频谱
        int              *dev_peak = NULL;          //各波束中的峰值点
        int              *dev_valley = NULL;        //各波束中的谷点
        cufftReal        *dev_preselected = NULL;   //预选的峰值点
        cufftReal        *dev_selected = NULL;      //筛选出的峰值点
        int              *peak = NULL;
        int              *valley = NULL;
        //-----------------Hilbert变换参数-------------------------------
        cufftComplex *dev_matchdatafreq=NULL;	//信号频响
        cudaStatus = cudaMalloc((void **)&dev_matchdatafreq, sizeof(cufftComplex)*NHILBT);
        cudaMemset((void **)&dev_matchdatafreq,0,sizeof(cufftComplex)*NHILBT);

        cufftComplex *dev_hilboutfreq=NULL;	//输出频响
        cudaStatus = cudaMalloc((void **)&dev_hilboutfreq, sizeof(cufftComplex)*NHILBT);
        cudaMemset((void **)&dev_hilboutfreq,0,sizeof(cufftComplex)*NHILBT);

        cufftReal *dev_hilbout=NULL;	//输出信号
        cudaStatus = cudaMalloc((void **)&dev_hilbout, sizeof(cufftReal)*FRAMELEN);
        cudaMemset((void **)&dev_hilbout,0,sizeof(cufftReal)*FRAMELEN);

        cufftReal *dev_envelopedata=NULL;//包络
        cudaStatus = cudaMalloc((void **)&dev_envelopedata, sizeof(cufftReal)*FRAMELEN);
        cudaMemset((void **)&dev_envelopedata,0,sizeof(cufftReal)*FRAMELEN);
        //----------------------------------------------------------------

        //-----------------调试：分配内存-----------------------------------
        cufftPlan1d(&Hplan, NFIR, CUFFT_R2C, 1);
        cufftPlan1d(&Xplan, NFIR, CUFFT_R2C, 1);
        cufftPlan1d(&Yplan, NFIR, CUFFT_C2R, 1);
        cufftPlan1d(&HXplan, NHILBT, CUFFT_R2C, 1);
        cufftPlan1d(&HYplan, FRAMELEN, CUFFT_C2R, 1);
        cufftPlan1d(&MXplan, NMAT, CUFFT_R2C, 1);
        cufftPlan1d(&MHplan, NMAT, CUFFT_R2C, 1);
        cufftPlan1d(&MYplan, NMAT, CUFFT_C2R, 1);

        cudaStatus = cudaMalloc((void **)&dev_dTime, sizeof(cufftReal)*(PHINUM*THETANUM*ELENUM));//将延时向量写入显存
        if (cudaStatus != cudaSuccess)
        {
                printf (" dev_dTime cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_dTime,0,sizeof(cufftReal)*(PHINUM*THETANUM*ELENUM));
        cudaMemcpy(dev_dTime,dTime,sizeof(cufftReal)*(PHINUM*THETANUM*ELENUM),cudaMemcpyHostToDevice);

        cudaStatus = cudaMalloc((void **)&dev_mat, sizeof(cufftReal)*NMAT);//将参考信号写入显存:频域
        if (cudaStatus != cudaSuccess)
        {
                printf (" dev_mat cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_mat,0,sizeof(cufftReal)*NMAT);
        cudaMemcpy(dev_mat+NMAT-NREF,rsRef,sizeof(cufftReal)*NREF,cudaMemcpyHostToDevice);//对其尾部

        cudaStatus = cudaMalloc((void **)&dev_fft_mat, sizeof(cufftComplex)*NMAT);//匹配滤波参数频谱
        if (cudaStatus != cudaSuccess)
        {
                printf (" dev_fft_mat cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_fft_mat,0,sizeof(cufftComplex)*NMAT);
        cufftExecR2C(MHplan,(cufftReal *)&dev_mat[0],(cufftComplex *)&dev_fft_mat[0]);

        cudaStatus = cudaMalloc((void **)&dev_fft_matout, sizeof(cufftComplex)*NMAT);//匹配滤波输出频谱
        if (cudaStatus != cudaSuccess)
        {
                printf (" dev_fft_matout cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_fft_matout,0,sizeof(cufftComplex)*NMAT);


        cudaStatus = cudaMalloc((void **)&dev_x, sizeof(cufftReal)*(FRAMELEN*ELENUM));//给原始数据分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf (" dev_x cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_x,0,sizeof(cufftReal)*FRAMELEN*ELENUM);
        cudaStatus = cudaMalloc((void **)&dev_x_s, sizeof(cufftReal)*NFIR);//给单个通道原始数据分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf (" dev_x_s cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_x_s,0,sizeof(cufftReal)*NFIR);
        cudaStatus = cudaMalloc((void **)&dev_h, sizeof(cufftReal)*NFIR);//给滤波器参数分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_h cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_h,0,sizeof(cufftReal)*NFIR);

        cudaStatus = cudaMalloc((void **)&dev_y, sizeof(cufftReal)*NFIR*ELENUM);//给滤波器输出数据分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_y cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_y,0,sizeof(cufftReal)*NFIR*ELENUM);

        cudaStatus = cudaMalloc((void **)&dev_fft_x,sizeof(cufftComplex)*NFIR*ELENUM);//给原始信号频域分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_fft_x cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_fft_x,0,sizeof(cufftComplex)*NFIR*ELENUM);

        cudaStatus = cudaMalloc((void **)&dev_fft_h,sizeof(cufftComplex)*NFIR);//给滤波器频域分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_fft_h cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_fft_h,0,sizeof(cufftComplex)*NFIR);

        cudaStatus = cudaMalloc((void **)&dev_fft_y,sizeof(cufftComplex)*(ELENUM*NFIR));//给输出数据频域分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_fft_y cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_fft_y,0,sizeof(cufftComplex)*(ELENUM*NFIR));

        cudaStatus = cudaMalloc((void **)&dev_chanbuff,sizeof(cufftReal)*FILTER_FRAME*ELENUM);//给通道缓存数据分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_chanbuff cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_chanbuff,0,sizeof(cufftReal)*FILTER_FRAME*ELENUM);


        fir1(FIRORDER,3,fl,fh,FS,5,h);
        cudaMemcpy(dev_h,h,sizeof(cufftReal)*FIRORDER,cudaMemcpyHostToDevice);
        cufftExecR2C(Hplan,(cufftReal *)&dev_h[0],(cufftComplex *)&dev_fft_h[0]);//得到滤波器频域响应dev_fft_h

        //---------------------------------------------------------------
        cudaStatus =cudaMalloc((void **)&dev_delayfilterbuf,sizeof(cufftReal)*FRAMELEN*ELENUM);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_delayfilterbuf cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_delayfilterbuf,0,sizeof(cufftReal)*FRAMELEN*ELENUM);


        cudaStatus =cudaMalloc((void **)&dev_delayfilterout,sizeof(cufftReal)*FRAMELEN*ELENUM);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_delayfilterout cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_delayfilterout,0,sizeof(cufftReal)*FRAMELEN*ELENUM);


        cudaStatus =cudaMalloc((void **)&dev_delayFilter,sizeof(cufftReal)*(2*M+1)*ELENUM);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_delayFilter cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_delayFilter,0,sizeof(cufftReal)*(2*M+1)*ELENUM);


        cudaStatus =cudaMalloc((void **)&dev_dI,sizeof(int)*ELENUM);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_dI cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_dI,0,sizeof(int)*ELENUM);


        cudaStatus =cudaMalloc((void **)&dev_dF,sizeof(cufftReal)*ELENUM);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_dF cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_dF,0,sizeof(cufftReal)*ELENUM);


        cudaStatus =cudaMalloc((void **)&dev_delaydata,sizeof(cufftReal)*FRAMELEN);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_delaydata cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_delaydata,0,sizeof(cufftReal)*FRAMELEN);


        cudaStatus =cudaMalloc((void **)&dev_matchdata,sizeof(cufftReal)*NMAT);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_matchdata cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_matchdata,0,sizeof(cufftReal)*NMAT);

        cudaStatus =cudaMalloc((void **)&dev_beamdata,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_beamdata cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_beamdata,0,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN);

        cudaStatus = cudaMalloc((void **)&dev_fft_delaydata,sizeof(cufftComplex)*NMAT);//给原始信号频域分配显存
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_fft_delaydata cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_fft_delaydata,0,sizeof(cufftComplex)*NMAT);

        cudaStatus = cudaMalloc((void **)&dev_peak,sizeof(int)*THETANUM*PHINUM*FRAMELEN);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_peak cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_peak,0,sizeof(int)*THETANUM*PHINUM*FRAMELEN);

        cudaStatus = cudaMalloc((void **)&dev_valley,sizeof(int)*THETANUM*PHINUM*FRAMELEN);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_valley cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_valley,0,sizeof(int)*THETANUM*PHINUM*FRAMELEN);

        cudaStatus = cudaMalloc((void **)&dev_preselected,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_preselected cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_preselected,0,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN);

        cudaStatus = cudaMalloc((void **)&dev_selected,sizeof(cufftReal)*THETANUM*PHINUM*4);
        if (cudaStatus != cudaSuccess)
        {
                printf ("dev_selected cudaMalloc Error! \n ");
        }
        cudaMemset((void **)&dev_selected,0,sizeof(cufftReal)*THETANUM*PHINUM*4);

        peak = (int *)malloc(THETANUM*PHINUM*FRAMELEN*sizeof(int));
        memset(peak,0,THETANUM*PHINUM*FRAMELEN*sizeof(int));

        valley = (int *)malloc(THETANUM*PHINUM*FRAMELEN*sizeof(int));
        memset(valley,0,THETANUM*PHINUM*FRAMELEN*sizeof(int));
        //-----------------调试：分配滤波器内存结束-----------------------------------

        //--------------------------测时延变量----------------------------
        cudaEvent_t start1;
        cudaEvent_t stop1;
        float msecTotal = 0.0f;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        //----------------------------------------------------------------
       

        while(1)
        { 
		FileName="/home/ubuntu/Documents/Active/tmp/beamdata"+std::to_string(fIndex++)+".bin";
        	wfp=fopen(FileName.c_str(),"wb");
		//printf("wait for process\n");
                pthread_mutex_lock(&count_lock_ActiveFrameDataReady);
                while (count_ActiveFrameDataReady == 0)
                {
                        pthread_cond_wait(&cond_ActiveFrameDataReady,&count_lock_ActiveFrameDataReady);
                }
                count_ActiveFrameDataReady = count_ActiveFrameDataReady -1;
                pthread_mutex_unlock(&count_lock_ActiveFrameDataReady);
                FrameNum++;

		cudaEventRecord(start1,NULL);
                cudaMemcpy(dev_x,ChannDataBuf,sizeof(cufftReal)*FRAMELEN*ELENUM,cudaMemcpyHostToDevice);//将ChannDataBuf数据写入dev_x
		
                //-----------------------------------------(1) 信号滤波(13ms)---------------------------------------------------
                //cudaEventRecord(start1,NULL);
                for(int jj=0;jj<ELENUM;jj++)
                {
                        cudaMemcpy(dev_x_s,dev_x+FRAMELEN*jj,sizeof(cufftReal)*FRAMELEN,cudaMemcpyDeviceToDevice);//将dev_x中第jj通道的数据移入到dev_x_s中
                        cufftExecR2C(Xplan,(cufftReal *)&dev_x_s[0],(cufftComplex *)&dev_fft_x[jj*NFIR]);//原信号傅立叶变换
                }

                //
                //频域相乘
                ActiveFilter<<<ELENUM*NFIR/THREADNUMPERBLK,THREADNUMPERBLK>>>(dev_fft_x,dev_fft_h,dev_fft_y,NFIR);

                //反变换
                for(int jj=0;jj<ELENUM;jj++)
                {
                        cufftExecC2R(Yplan,dev_fft_y+jj*NFIR,dev_y+jj*NFIR);
                        cudaMemcpy((float*)&dev_chanbuff[jj*FILTER_FRAME],(cufftReal*)&dev_y[jj*NFIR+FIRORDER/2],sizeof(float)*FILTER_FRAME,cudaMemcpyDeviceToDevice);
                }

                //QueryPerformanceCounter(&nEndTime);
                //cudaEventRecord(stop1,NULL);
                //cudaEventSynchronize(stop1);                
		//cudaEventElapsedTime(&msecTotal,start1,stop1);		
		//printf("%d:%.3f:\n",FrameNum,msecTotal);

                //-----------------------------------------(1) 信号滤波结束---------------------------------------------------

                //-----------------------------------------(2) 波束形成(160ms)---------------------------------------------------
                //cudaEventRecord(start1,NULL);
                //求延时后信号
                for(int ii=0;ii<THETANUM;ii++)
                {//俯仰角
                        for(int jj=0;jj<PHINUM;jj++)
                        {//方位角
                                int index=(ii*9+jj)*12;//时延标号
                                ActiveDelayFilterGen<<<ELENUM,2*M+1>>>(dev_delayFilter,dev_dI,dev_dF,dev_dTime,index);
                                for (int kk=0;kk<12;kk++)
                                {
                                        int DI=(int)dTime[index+kk];//整数时延
                                        float DF=dTime[index+kk];
                                        if(DI>=0)
                                        {
                                                cudaMemcpy(dev_delayfilterbuf+kk*FRAMELEN+DI,dev_chanbuff+kk*FRAMELEN,sizeof(cufftReal)*(FRAMELEN-DI),cudaMemcpyDeviceToDevice);
                                        }
                                        else
                                        {
                                                cudaMemcpy(dev_delayfilterbuf+kk*FRAMELEN,dev_chanbuff+kk*FRAMELEN-DI,sizeof(cufftReal)*(FRAMELEN+DI),cudaMemcpyDeviceToDevice);
                                        }
                                        if(DF > 0.0001)
                                        {
                                                ActiveFineDelayFilter<<<FRAMELEN,2*M+1>>>(dev_delayfilterbuf+kk*FRAMELEN,dev_delayfilterout+kk*FRAMELEN,dev_delayFilter+kk*(2*M+1));
                                                //cudaMemcpy(dev_delayfilterout+kk*FRAMELEN,dev_delayfilterbuf+kk*FRAMELEN,sizeof(cufftReal)*FRAMELEN,cudaMemcpyDeviceToDevice);
                                        }
                                        else
                                        {
                                                cudaMemcpy(dev_delayfilterout+kk*FRAMELEN,dev_delayfilterbuf+kk*FRAMELEN,sizeof(cufftReal)*FRAMELEN,cudaMemcpyDeviceToDevice);
                                        }
                                }
                                MatrixSumRow<<<FRAMELEN,1>>>(dev_delayfilterout,dev_delaydata,ELENUM,FRAMELEN);
                                //匹配滤波
                                //==============频域匹配滤波============================
                                cufftExecR2C(MXplan,dev_delaydata,dev_fft_delaydata);
                                VectorMultiplier<<<NMAT,1>>>(dev_fft_delaydata,dev_fft_mat,dev_fft_matout);
                                cufftExecC2R(MYplan,dev_fft_matout,dev_matchdata);

                                //Hilbert变换取包络(0.5ms)
                                //1.原始信号傅里叶变换   dev_matchdata -> dev_matchdatafreq
                                cufftExecR2C(HXplan,dev_matchdata,dev_matchdatafreq);
                                //2.频域变换  dev_matchdatafreq -> dev_hilboutfreq
                                HilbFilt<<<NHILBT,1>>>(dev_hilboutfreq, dev_matchdatafreq, NHILBT/2);
                                //3.逆傅里叶变换  dev_hilboutfreq -> dev_hilbout
                                cufftExecC2R(HYplan,dev_hilboutfreq,dev_hilbout);
                                //4.求包络 dev_matchdata,dev_hilbout ->dev_envelopedata
                                Envelope<<<FRAMELEN,1>>>( dev_envelopedata,dev_matchdata, dev_hilbout);

                                //数据存入 dev_beamdata
                                cudaMemcpy((cufftReal *)&dev_beamdata[(ii*9+jj)*FRAMELEN],dev_envelopedata,sizeof(cufftReal)*FRAMELEN,cudaMemcpyDeviceToDevice);
                        }
                }
		//cudaEventRecord(start1,NULL);
		//90ms
                cudaMemset((void **)&dev_peak,0,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN);
                cudaMemset((void **)&dev_valley,0,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN);
               	DevFindPeak<<<PHINUM*THETANUM,1>>>(dev_beamdata,dev_peak,FRAMELEN);
                cudaMemcpy(peak,dev_peak,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN,cudaMemcpyDeviceToHost);
             	DevFindValley<<<PHINUM*THETANUM,1>>>(dev_beamdata,dev_valley,FRAMELEN);
                cudaMemcpy(valley,dev_valley,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN,cudaMemcpyDeviceToHost);
             DevPeakDetection<<<PHINUM*THETANUM,1>>>(dev_peak,dev_valley,dev_beamdata,dev_preselected,dev_selected,FRAMELEN,3.0,10000);

		cudaEventRecord(stop1,NULL);
                cudaEventSynchronize(stop1);                
		cudaEventElapsedTime(&msecTotal,start1,stop1);		
		printf("%d:%.3f:\n",FrameNum,msecTotal);

		cudaMemcpy(selected,dev_selected,sizeof(cufftReal)*THETANUM*PHINUM*4,cudaMemcpyDeviceToHost);
		int Imax=0;  
		float Vmax=0.0;
		float Dmax=0.0;
		for(int i=0;i<THETANUM*PHINUM;i++){
			if(selected[i*4+3]>Vmax){
				Vmax=selected[i*4+3];
				Imax=i;
			}
		}
		Dmax=selected[Imax*4+2];
		printf("#%d : %.1f\n",Imax,Dmax);
                //cudaEventRecord(stop1,NULL);
                //cudaEventSynchronize(stop1);

                cudaMemcpy(beamdata,dev_beamdata,sizeof(cufftReal)*THETANUM*PHINUM*FRAMELEN,cudaMemcpyDeviceToHost);//

                cudaEventElapsedTime(&msecTotal,start1,stop1);		
		printf("%d:%.3f:\n",FrameNum,msecTotal);

		cudaMemcpy(temp,dev_beamdata,sizeof(cufftReal)*6800*27,cudaMemcpyDeviceToHost);
                fwrite(temp,sizeof(float),6800*27,wfp);			
		
                //printf("%d:%.3f:\n",FrameNum,msecTotal);
		printf("processing finished.\n");
        	fclose(wfp);
        	wfp=NULL;
        //-----------------------------------------(2) 波束形成结束-----------------------------------------------
        }
}

void *ActiveReceiveNetwork(void *lParam)
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
        int CounterA_B1 = FRAMELEN;
	int CounterA=FRAMELEN;
        int temp = 0;
        int FrameNum1 = 0,FrameNum2 = 0, FrameNum = 0;
        bool foundpulse = false;
        int num=0;

        if(DataBufA_B1 != NULL)
        {
            free(DataBufA_B1);
            DataBufA_B1 = NULL;
       }
      DataBufA_B1 = (int *)malloc(FRAMELEN*CHANNUM*sizeof(int));
      memset(DataBufA_B1,0,FRAMELEN*CHANNUM*sizeof(int));

 
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
                                 if(sourceid == 0)
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
//                                if(sourceid == 2)
//                                {
//                                      FrameNum2++;
//                                      memcpy(readbufb2,pktStr+42,(TL*CHANNUM+1)*sizeof(int));
//                                      FramenumN2 = *(pktStr+44);
//                                      FramenumN2 = FramenumN2 >> 2;
//                                      if (FrameNum2 == 1)
//                                      {
//                                            LastFramenumN2 = FramenumN2;
//                                      }
//                                      else
//                                      {
//                                            if (FramenumN2 != LastFramenumN2+1 && FramenumN2+63 != LastFramenumN2)
//                                            {
//                                                  printf("Lost Board2 data package!\n");
//                                            }
//                                            LastFramenumN2 = FramenumN2;
//                                      }
//                                }
                                 //搜索脉冲前沿
                                 if(!foundpulse)
                                 {
                                         float dataval = 0.0;
                                         for(int kk=0;kk<TL;kk++)
                                         {
                                                 temp = readbufb1[3+kk*CHANNUM+1];//2通道
                                                 temp = temp<<8;
                                                 temp = temp>>8;
                                                 dataval = temp*1.0/pow(2.0,23) * 2.5;
                                                 if(fabs(dataval) > 0.5)
                                                 {
                                                         foundpulse = true;
                                                         break;
                                                 }
                                         }
                                 }
                                 if(foundpulse && num++ > DIRECTARRIVENUM) //去掉直达波后510个点  510/17=30
                                 {
                                         memcpy(DataBufA_B1+FRAMELEN*CHANNUM-CounterA*CHANNUM,readbufb1+3,TL*CHANNUM*sizeof(int));
                                         CounterA = CounterA-TL;
                                         if(CounterA == 0)
                                         {
                                                 //使事件有效
                                                 pthread_mutex_lock(&count_lock_Board0DataReady);
                                                 pthread_cond_signal(&cond_Board0DataReady);
                                                 count_Board0DataReady = count_Board0DataReady+1;
                                                 pthread_mutex_unlock(&count_lock_Board0DataReady);
                                                 foundpulse = false;
                                                 CounterA = FRAMELEN;
                                                 num=0;
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












