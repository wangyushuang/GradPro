%% 2018/3/12
% BeamFormer�Ľ��棬�Ľ�����
% 1.����ˮ����ͨ����λУ׼DataCorrect
% 2.�ı�����ϵ
cd F:\2018\March\3_12
TestMain2
DataCorrect; %У׼ˮ������λ
close all;
clearvars -except data
%% �źŲ���
fs=250e3; %������
f=95e3; %�ź�Ƶ�ʣ�90kHz-100kHz
c=1500; %����
T=size(data,1)/fs; %����ʱ��
Rc1=23e-3; %�⾶:23mm
Rc2=11.5e-3; %�ھ�:11.5mm
N1=6; %��Ȧ��Ԫ����
N2=6; %��Ȧ��Ԫ����
% ˮ��������λ��xOzƽ��
xEle=[Rc1*sin([0:N1-1]*2*pi/N1),Rc2*sin([0:N2-1]*2*pi/N2)]'; %��Ԫx����
zEle=[Rc1*cos([0:N1-1]*2*pi/N1),Rc2*cos([0:N2-1]*2*pi/N2)]'; %��Ԫz����
%% �˲�
 %���룺data �����yout1
 N=128; %����
 win=hamming(N);
 bbp = fir1(N-1,[80E3 120E3]/fs*2,'bandpass',win,'scale');
 yout1=zeros(size(data,1),12);
 for k=1:12
     yout1(:,k)=filter(bbp,1,data(:,k));
 end

%% ����Ԥ����:��ȡ��Ч����
% ���룺yout1 �����yout2
% ��ȡ1s���źţ����ֱ�ﲨ��ʼλ��
 p1s=round(fs); %1s���ε���
 p1ms=round(fs*1e-3); %1ms���ε���
 Tmsec=13; %��ȡʱ�䳤�ȣ�13ms ��Ӧ��Զ����Ϊ10m
 pseg=round(p1ms*Tmsec); %��ȡ���ε��� points of a segment
 yout2=zeros(pseg,12);
 pos=zeros(12,1);
 for ch=1:12
     Vmax=max(yout1(1:p1s,ch));%��ͨ���ڽ�ȡ����������������
     for k=1:p1s %���ؼ��
        if(yout1(k)>0.2*Vmax)
            pos(ch)=k;%����ֱ�ﲨ��ʼλ��
            break;
        end
     end
 end
 staPos=round(mean(pos));%ֱ�ﲨ��ʼ��
 endPos=staPos+round(p1ms*1.7);%ֱ�ﲨ������
 for ch=1:12
     for h=staPos:endPos  %ä��
         yout2(h-staPos+1,ch)=0;
     end
     for h=(endPos+1):(endPos+pseg) %��Ч����Ƭ��
         yout2(h-staPos+1,ch)=yout1(h,ch);
     end
 end
 %������ͼ
 figure(1)
 plot(yout2(:,:),'.-') %������Ĳ���
 title('��ȡƬ��')
 
 %% �����γ���ƥ���˲�
 % Hilbert�任
 % ���룺yout2 �����yc
 yc=zeros(size(yout2));
 for ch=1:12
        yc(:,ch)=hilbert(yout2(:,ch));
 end
 R=yc.'*conj(yc);
 t=linspace(0,1e-3,p1ms);
 sRef=sin(2*pi*(90e3*t+0.5e7*t.^2));%�ο��ź�
 theta=[-40:10:40]*pi/180+pi/2;
 phy=[-10:10:10]*pi/180;
 P=zeros(length(theta),length(phy));
 Pmax=0;
 for th=1:length(theta)
     for ph=1:length(phy)
         r=xEle*cos(phy(ph))*cos(theta(th))+zEle*sin(phy(ph));%���̲�  
         w=exp(-1j*2*pi*f*r/c);
         P1(th,ph)=w'*R*w;
         ybm=w'*yc.';% �������
         ymf=MatchedFilter(real(ybm),sRef);
         P(th,ph)=abs(ymf'*ymf);
         %��¼�������㴦��Ϣ
         if(P(th,ph)>Pmax)
                Pmax=P(th,ph);
                wmax=w;
         end
     end
 end
 %������󴦵�ƥ���˲������
 ymax=MatchedFilter(real(wmax'*yc.'),sRef);
 figure
 dist=[1:length(ymax)]/fs*c/2; %����
 plot(dist,ymax)
 xlabel('����/��')
 figure
 imagesc(theta*180/pi-90,phy*180/pi,P')
 xlabel('��λ��')
 ylabel('������')







