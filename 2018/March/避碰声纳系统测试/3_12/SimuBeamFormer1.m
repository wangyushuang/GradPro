%% �������źŲ���
% ģ��ʵ���������źŲ����γ�Ч��
% Simu_BeamFormer�Ľ��棺�任����ϵ
close all;
% clear;
 fs=250e3;%������
 f=95e3;%�ź�����Ƶ�ʣ�90kHz-100kHz
 c=1500;%����
 Rc1=0.023;%�⾶
 Rc2=0.0115;%�ھ�
 N1=6;%��Ȧ��Ԫ����
 N2=6;%��Ȧ��Ԫ����
xEle=[Rc1*sin([0:N1-1]*2*pi/N1),Rc2*sin([0:N2-1]*2*pi/N2)]'; %��Ԫx����
zEle=[Rc1*cos([0:N1-1]*2*pi/N1),Rc2*cos([0:N2-1]*2*pi/N2)]'; %��Ԫz����
 
 phy0=10*pi/180;%�ź����丩����
 theta0=-30*pi/180+pi/2;%�ź����䷽λ��
 T=0.01;%����ʱ��
 t=linspace(0,T,T*fs);
 r0=xEle*cos(phy0)*cos(theta0)+zEle*sin(phy0);%ʵ�ʲ��̲�
 s=exp(-1j*2*pi*(90e3*t+0.5e7*t.^2));%LFM�ź�
 sout=exp(-1j*2*pi*f*r0/c)*s;
 sout=sout+wgn(size(sout,1),size(sout,2),0.01,'complex');
 R=sout*sout';
 %% �����γ�
 theta=linspace(-40,40,100)*pi/180+pi/2;
 phy=linspace(-10,10,100)*pi/180;
 P=zeros(length(theta),length(phy));
 for i=1:length(theta)
    for k=1:length(phy)
        r=xEle*cos(phy(k))*cos(theta(i))+zEle*sin(phy(k));%���̲�
        w=exp(-1j*2*pi*f*r/c);%
        P(i,k)=w'*R*w;%1/(w'/(R+0.5*eye(12))*w);
    end
 end
P=real(P)/max(max(real(P)));
imagesc(theta*180/pi-90,phy*180/pi,P')
xlabel('��λ��')
ylabel('������')




