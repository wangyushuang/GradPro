%�����γ�
% cd F:\2018\March\Data\data3_9\bp
% TestMain2
close all;
clearvars -except data
%�źŲ���
 fs=250e3;%������
 f=95e3;%�ź�Ƶ�ʣ�90kHz-100kHz
 c=1500;%����
 T=size(data,1)/fs;%����ʱ��
 Rc1=0.023;%�⾶
 Rc2=0.0115;%�ھ�
 N1=6;%��Ȧ��Ԫ����
 N2=6;%��Ȧ��Ԫ����
 xEle=[Rc1*cos([0:N1-1]*pi/3),Rc2*cos([0:N2-1]*pi/3)]';%��Ԫx����
 yEle=[Rc1*sin([0:N1-1]*pi/3),Rc2*sin([0:N2-1]*pi/3)]';%��Ԫy����
%% ������
%  Nus=1;%����������
%  Nds=1;%����������
% %  yout=zeros(size(data,1)*Nus,13);%�˲������
%  for k=1:13
%      yout(:,k)=resample(data(:,k),Nus,Nds);
%  end

 %% �˲�
 %���룺data �����yout1
 N=120;%����
 win=hamming(N);
 bbp = fir1(N-1,[80E3 120E3]/fs*2,'bandpass',win,'scale');
 for k=1:13
     yout1(:,k)=filter(bbp,1,data(:,k));
 end
%  plot(yout(:,1))
%  figure;
%  plot(yout1(:,1),'.-')
%% ����Ԥ����:ȥ��ֱ�ﲨ(����һ������)
%���룺yout1 �����yout2
 p1s=fs;%1s���ε���
 p1ms=round(fs*1e-3);%1ms���ε���
 Tsec=1.8;%��ȡʱ�䳤�ȣ���λ-s
 psec=round(p1s*Tsec);%��ȡ���ε���
 yout2=yout1(1:psec,:);
 pos=zeros(12,1);
 for ch=1:12
     Vmax=max(yout1(1:psec,ch));%��ͨ���ڽ�ȡ����������������
     for k=1:size(data,1)%���ؼ��
        if(yout1(k)>0.4*Vmax)
            pos(ch)=k;%����ֱ�ﲨ��ʼλ��
            break;
        end
     end
     for h=max(1,pos-50):(pos+round(p1ms*1)) %����ʼλ��ǰ50���㵽��1.5ms����������
         yout2(h,ch)=0;
     end
 end
 %������ͼ
 figure(1)
 subplot 311
 plot(data(1:psec),'.-') %ԭʼ����
 title('ԭʼ����')
 subplot 312
 plot(yout1(1:psec),'.-') %�˲����������
 title('�˲����������')
 subplot 313
 plot(yout2(1:psec),'.-') %ȥ��ֱ�ﲨ����
 title('ȥ��ֱ�ﲨ����')

 %% �����γ�: 1.5�ף�1ms��һ��,��mean(pos)��15m(��0.1s)
  % Hilber�任
 %���룺yout2 �����yc 
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
    % dataRef=hilbert(yout1(:,13));%�ο�ͨ��
    R=(yc(:,1:12))'*(yc(:,1:12));%Э�������

    for i=1:length(theta)
        for k=1:length(phy)
            r=xEle*sin(phy(k))*cos(theta(i))+yEle*sin(phy(k))*sin(theta(i));%���̲�  phy=0deg��ʾָ����ǰ��
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
     xlabel('��λ��')
     ylabel('������')
end
%% �����γ����
t=linspace(0,1e-3,p1ms);
sRef=cos(2*pi*(90e3*t+0.5e7*t.^2));%�ο��ź�
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
title('0�ȷ��������')
subplot 212
plot(result)
title('ƥ���˲������')
 
 
 
 
 
 
 