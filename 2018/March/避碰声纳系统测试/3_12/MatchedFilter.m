function yout=MatchedFilter(s, ref)
% s-�ź�
%ref-�ο��ź�
Ns=length(s);
Nr=length(ref);
yout=zeros(Ns,1);
% ��Ҫ��֤NrΪ����
if(rem(Nr,2)==0) %���Ϊż������ȥ�����һ����
    Nr=Nr-1;
end
Nrh=(Nr+1)/2;%�ο��ź��е�
for k=1:Ns
%     for h=max(1,Nrh-k+1):min(Nr,Nrh+Ns-k)    %ѭ�����´���ʱ�䳤    
%         yout(k)=yout(k)+ref(h)*s(k-Nrh+h);
%     end    
    range=[max(1,Nrh-k+1):min(Nr,Nrh+Ns-k)];
    yout(k)=ref(range)*s(k-Nrh+range)'; %���þ���˷�����ڲ�ѭ��
end

%% ���԰���
% t=linspace(0,1,1000);
% f=10e3;
% ref=sin(2*pi*f*t);
% s(1:1000)=0;
% s(1001:2000)=ref;
% s(2001:3000)=0;
% yout=MatchedFilter(s,ref);
% subplot 311
% plot(ref);
% title('�ο��ź�')
% subplot 312 
% plot(s)
% title('�����ź�')
% subplot 313
% plot(yout)
% title('����ź�')


