%波束形成
% cd F:\2018\March\Data\data3_9\bp
% TestMain2
close all;
clearvars -except data
%信号参数
 fs=250e3;%采样率
 f=95e3;%信号频率：90kHz-100kHz
 c=1500;%声速
 T=size(data,1)/fs;%采样时间
 Rc1=0.023;%外径
 Rc2=0.0115;%内径
 N1=6;%外圈阵元个数
 N2=6;%内圈阵元个数
 xEle=[Rc1*cos([0:N1-1]*pi/3),Rc2*cos([0:N2-1]*pi/3)]';%阵元x坐标
 yEle=[Rc1*sin([0:N1-1]*pi/3),Rc2*sin([0:N2-1]*pi/3)]';%阵元y坐标
%% 升采样
%  Nus=1;%升采样倍数
%  Nds=1;%降采样倍数
% %  yout=zeros(size(data,1)*Nus,13);%滤波器输出
%  for k=1:13
%      yout(:,k)=resample(data(:,k),Nus,Nds);
%  end

 %% 滤波
 %输入：data 输出：yout1
 N=120;%阶数
 win=hamming(N);
 bbp = fir1(N-1,[80E3 120E3]/fs*2,'bandpass',win,'scale');
 for k=1:13
     yout1(:,k)=filter(bbp,1,data(:,k));
 end
%  plot(yout(:,1))
%  figure;
%  plot(yout1(:,1),'.-')
%% 波形预处理:去除直达波(处理一个周期)
%输入：yout1 输出：yout2
 p1s=fs;%1s波形点数
 p1ms=round(fs*1e-3);%1ms波形点数
 Tsec=1.8;%截取时间长度：单位-s
 psec=round(p1s*Tsec);%截取波形点数
 yout2=yout1(1:psec,:);
 pos=zeros(12,1);
 for ch=1:12
     Vmax=max(yout1(1:psec,ch));%该通道在截取长度内最大输出幅度
     for k=1:size(data,1)%边沿检测
        if(yout1(k)>0.4*Vmax)
            pos(ch)=k;%记下直达波起始位置
            break;
        end
     end
     for h=max(1,pos-50):(pos+round(p1ms*1)) %将起始位置前50个点到后1.5ms的数据置零
         yout2(h,ch)=0;
     end
 end
 %画波形图
 figure(1)
 subplot 311
 plot(data(1:psec),'.-') %原始波形
 title('原始波形')
 subplot 312
 plot(yout1(1:psec),'.-') %滤波器输出波形
 title('滤波器输出波形')
 subplot 313
 plot(yout2(1:psec),'.-') %去掉直达波波形
 title('去掉直达波波形')

 %% 波束形成: 1.5米（1ms）一段,从mean(pos)到15m(即0.1s)
  % Hilber变换
 %输入：yout2 输出：yc 
 theta=linspace(0,2*pi,361);
 phy=linspace(0,pi/2,91);
 P=zeros(length(theta),length(phy));
 P_db=zeros(length(theta),length(phy));

 pstart=mean(pos);
 pdelta=round(0.1*p1s);
for h=1:9
    for ch=1:12
        yc(:,ch)=hilbert(yout2(pstart+(h-1)*pdelta:pstart+h*pdelta,ch));
    end
    % dataRef=hilbert(yout1(:,13));%参考通道
    R=(yc(:,1:12))'*(yc(:,1:12));%协方差矩阵

    for i=1:length(theta)
        for k=1:length(phy)
            r=xEle*sin(phy(k))*cos(theta(i))+yEle*sin(phy(k))*sin(theta(i));%波程差  phy=0deg表示指向正前方
            w=exp(-1j*2*pi*f*r/c);%
            P(i,k,h)=w'*R*w;%1/(w'/(R+0.5*eye(12))*w);
        end
    end
    P(:,:,h)=abs(P(:,:,h))/max(max(abs(P(:,:,h))));
    P_db(:,:,h)=10*log10(P(:,:,h));
end
 for h=1:5
    P(:,:,h)=abs(P(:,:,h))/max(max(max(abs(P))));
    P_db(:,:,h)=10*log10(P(:,:,h));
 end
 [mthe,mphy]=meshgrid(theta,phy);
 figure
%  mesh(mthe*180/pi,mphy*180/pi,P(:,:,1)')
for h=1:9
    subplot(3,3,h)
    imagesc(theta*180/pi,phy*180/pi,P(:,:,h)')
     xlabel('方位角')
     ylabel('俯仰角')
end
%% 波束形成输出
t=linspace(0,1e-3,p1ms);
sRef=cos(2*pi*(90e3*t+0.5e7*t.^2));%参考信号
ysum=sum(yout2(:,1:12),2);
iLen=length(sRef);
result=zeros(size(ysum));
for k=1:length(ysum)
    for h=max(1,(k-iLen+1)):k
        result(k)=result(k)+ysum(h)*sRef(h-k+iLen);
    end
end 
figure
subplot 211
plot(ysum)
title('0度方向波束输出')
subplot 212
plot(result)
title('匹配滤波器输出')
 
 
 
 
 
 
 