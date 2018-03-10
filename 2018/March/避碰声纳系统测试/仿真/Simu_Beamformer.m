%% 阵列与信号参数
% 模拟实际阵列与信号波束形成效果
close all;
% clear;
 fs=250e3;%采样率
 f=95e3;%信号中心频率：90kHz-100kHz
 c=1500;%声速
 Rc1=0.023;%外径
 Rc2=0.0115;%内径
 N1=6;%外圈阵元个数
 N2=6;%内圈阵元个数
 xEle=[Rc1*cos(-[0:N1-1]*pi/3),Rc2*cos(-[0:N2-1]*pi/3)]';%阵元x坐标
 yEle=[Rc1*sin(-[0:N1-1]*pi/3),Rc2*sin(-[0:N2-1]*pi/3)]';%阵元y坐标
 
 phy0=10*pi/180;%信号入射方位角
 theta0=100*pi/180;%信号入射俯仰角
 T=0.01;%采样时间
 t=linspace(0,T,T*fs);
 r0=xEle*sin(phy0)*cos(theta0)+yEle*sin(phy0)*sin(theta0);%实际波程差
 s=exp(-1j*2*pi*f*t);%exp(-1j*2*pi*(90e3*t+0.5e7*t.^2));%LFM信号
 sout=exp(-1j*2*pi*f*r0/c)*s;
 sout=sout+wgn(size(sout,1),size(sout,2),0.01,'complex');
 R=sout*sout';
 %% 波束形成
 theta=linspace(0,2*pi,361);
 phy=linspace(0,pi/2,181);
 P=zeros(length(theta),length(phy));
 for i=1:length(theta)
    for k=1:length(phy)
        r=xEle*sin(phy(k))*cos(theta(i))+yEle*sin(phy(k))*sin(theta(i));%波程差
        w=exp(-1j*2*pi*f*r/c);%
        P(i,k)=w'*R*w;%1/(w'/(R+0.5*eye(12))*w);
    end
 end
P=real(P)/max(max(real(P)));
imagesc(theta*180/pi,phy*180/pi,P')




