%% 2018/3/12
% BeamFormer改进版，改进内容
% 1.加入水听器通道相位校准DataCorrect
% 2.改变坐标系
cd F:\2018\March\3_12
TestMain2
DataCorrect; %校准水听器相位
close all;
clearvars -except data
%% 信号参数
fs=250e3; %采样率
f=95e3; %信号频率：90kHz-100kHz
c=1500; %声速
T=size(data,1)/fs; %采样时间
Rc1=23e-3; %外径:23mm
Rc2=11.5e-3; %内径:11.5mm
N1=6; %外圈阵元个数
N2=6; %内圈阵元个数
% 水听器阵列位于xOz平面
xEle=[Rc1*sin([0:N1-1]*2*pi/N1),Rc2*sin([0:N2-1]*2*pi/N2)]'; %阵元x坐标
zEle=[Rc1*cos([0:N1-1]*2*pi/N1),Rc2*cos([0:N2-1]*2*pi/N2)]'; %阵元z坐标
%% 滤波
 %输入：data 输出：yout1
 N=128; %阶数
 win=hamming(N);
 bbp = fir1(N-1,[80E3 120E3]/fs*2,'bandpass',win,'scale');
 yout1=zeros(size(data,1),12);
 for k=1:12
     yout1(:,k)=filter(bbp,1,data(:,k));
 end

%% 波形预处理:截取有效波形
% 输入：yout1 输出：yout2
% 先取1s的信号，检测直达波起始位置
 p1s=round(fs); %1s波形点数
 p1ms=round(fs*1e-3); %1ms波形点数
 Tmsec=13; %截取时间长度：13ms 对应最远距离为10m
 pseg=round(p1ms*Tmsec); %截取波形点数 points of a segment
 yout2=zeros(pseg,12);
 pos=zeros(12,1);
 for ch=1:12
     Vmax=max(yout1(1:p1s,ch));%该通道在截取长度内最大输出幅度
     for k=1:p1s %边沿检测
        if(yout1(k)>0.2*Vmax)
            pos(ch)=k;%记下直达波起始位置
            break;
        end
     end
 end
 staPos=round(mean(pos));%直达波起始点
 endPos=staPos+round(p1ms*1.7);%直达波结束点
 for ch=1:12
     for h=staPos:endPos  %盲区
         yout2(h-staPos+1,ch)=0;
     end
     for h=(endPos+1):(endPos+pseg) %有效波形片段
         yout2(h-staPos+1,ch)=yout1(h,ch);
     end
 end
 %画波形图
 figure(1)
 plot(yout2(:,:),'.-') %待处理的波形
 title('截取片段')
 
 %% 波束形成与匹配滤波
 % Hilbert变换
 % 输入：yout2 输出：yc
 yc=zeros(size(yout2));
 for ch=1:12
        yc(:,ch)=hilbert(yout2(:,ch));
 end
 R=yc.'*conj(yc);
 t=linspace(0,1e-3,p1ms);
 sRef=sin(2*pi*(90e3*t+0.5e7*t.^2));%参考信号
 theta=[-40:10:40]*pi/180+pi/2;
 phy=[-10:10:10]*pi/180;
 P=zeros(length(theta),length(phy));
 Pmax=0;
 for th=1:length(theta)
     for ph=1:length(phy)
         r=xEle*cos(phy(ph))*cos(theta(th))+zEle*sin(phy(ph));%波程差  
         w=exp(-1j*2*pi*f*r/c);
         P1(th,ph)=w'*R*w;
         ybm=w'*yc.';% 波束输出
         ymf=MatchedFilter(real(ybm),sRef);
         P(th,ph)=abs(ymf'*ymf);
         %记录能量最大点处信息
         if(P(th,ph)>Pmax)
                Pmax=P(th,ph);
                wmax=w;
         end
     end
 end
 %能量最大处的匹配滤波器输出
 ymax=MatchedFilter(real(wmax'*yc.'),sRef);
 figure
 dist=[1:length(ymax)]/fs*c/2; %距离
 plot(dist,ymax)
 xlabel('距离/米')
 figure
 imagesc(theta*180/pi-90,phy*180/pi,P')
 xlabel('方位角')
 ylabel('俯仰角')







