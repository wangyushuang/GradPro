close all;clear;
fid=fopen('F:\2018\March\Data\test3.bin');
A=fread(fid,'float');
fclose(fid);
f=87.5e3;%�ź�Ƶ��
fs=200e3;%������
ts=10;%����ʱ��
c=1500;%����
pChan=size(A,1)/13;%ÿͨ������
rdata=zeros(pChan,13);
% rs_rdata=zeros(pChan,13);
figure 
hold on
col=['r','k','r','k','r','k','r','k','r','k','r','k','g'];
for i=10
    rdata(1:pChan,i)=A((i-1)*pChan+1:i*pChan);
    plot([(i-1)*pChan+1:i*pChan],rdata(:,i),col(i))%ԭʼ����
    tmp=rdata(1:pChan,i);
    rs_rdata(:,i)=resample(tmp,10,1);
    tmp=rs_rdata(:,i);
    figure
    plot(rs_rdata(:,i),col(i))%����������
    N=100;%����
    win=hamming(N+1);
    bbp = fir1(N,[0.8 0.99]/10,'bandpass',win,'scale');
    outbp = filter(bbp,1,tmp);
    figure 
    plot(outbp)
end