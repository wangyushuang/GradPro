%% �������źŲ���
% ģ��ʵ���������źŲ����γ�Ч��
close all;
% clear;
 fs=250e3;%������
 f=95e3;%�ź�����Ƶ�ʣ�90kHz-100kHz
 c=1500;%����
 Rc1=0.023;%�⾶
 Rc2=0.0115;%�ھ�
 N1=6;%��Ȧ��Ԫ����
 N2=6;%��Ȧ��Ԫ����
 xEle=[Rc1*cos(-[0:N1-1]*pi/3),Rc2*cos(-[0:N2-1]*pi/3)]';%��Ԫx����
 yEle=[Rc1*sin(-[0:N1-1]*pi/3),Rc2*sin(-[0:N2-1]*pi/3)]';%��Ԫy����
 
 phy0=10*pi/180;%�ź����䷽λ��
 theta0=100*pi/180;%�ź����丩����
 T=0.01;%����ʱ��
 t=linspace(0,T,T*fs);
 r0=xEle*sin(phy0)*cos(theta0)+yEle*sin(phy0)*sin(theta0);%ʵ�ʲ��̲�
 s=exp(-1j*2*pi*f*t);%exp(-1j*2*pi*(90e3*t+0.5e7*t.^2));%LFM�ź�
 sout=exp(-1j*2*pi*f*r0/c)*s;
 sout=sout+wgn(size(sout,1),size(sout,2),0.01,'complex');
 R=sout*sout';
 %% �����γ�
 theta=linspace(0,2*pi,361);
 phy=linspace(0,pi/2,181);
 P=zeros(length(theta),length(phy));
 for i=1:length(theta)
    for k=1:length(phy)
        r=xEle*sin(phy(k))*cos(theta(i))+yEle*sin(phy(k))*sin(theta(i));%���̲�
        w=exp(-1j*2*pi*f*r/c);%
        P(i,k)=w'*R*w;%1/(w'/(R+0.5*eye(12))*w);
    end
 end
P=real(P)/max(max(real(P)));
imagesc(theta*180/pi,phy*180/pi,P')




