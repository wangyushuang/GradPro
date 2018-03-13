%Test noise,dynamic,interferrence,coherence
% ['1.Noise Test' 10 13 '2.Dynamic Test' 10 13 '3.Interferrence Test' 10 13 '4.Coherence Test' 10 13 '5.Sync Test' ]

function Test_Noise3(mode,from,to,FS,CH,PackPoint,pname,fname,gain)

    savefile = 'tmp.mat';
    
    data2=[];
    data=[];
    dataidx2=[];
    max_idx2=[];
    filecnt=1;
    FilePackNum=4000*20;
    Np=PackPoint*FilePackNum;
    for kk = from:to
        if(kk>9)
        fname(7:end+1)=fname(6:end);
        fname(6:7)=num2str(kk);
        else
        fname(6)=num2str(kk);
        end
        filename = [pname,fname]; 
        fid=fopen(filename,'rb');
        udpdata=fread(fid,[PackPoint*16+3,FilePackNum],'int32');
        fclose(fid);
        
        data16=reshape(udpdata(4:PackPoint*16+3,:),PackPoint*16*FilePackNum,1);
        
        filename_tmp=[filename,'.tmp'];
        fid=fopen(filename_tmp,'wb');
        fwrite(fid,data16,'int32');        
        fclose(fid);
        
        fid=fopen(filename_tmp,'rb');        
        [ch,ch_len]=fread(fid,'uint8');
        fclose(fid);
        
        delete(filename_tmp);

        ch_num = reshape(ch(4:4:ch_len),1,PackPoint*FilePackNum*16); % channel ID
        for ii=4:4:ch_len
            ch(ii)=0;
        end
        
        filename_tmp=[filename,'.tmp'];
        fid=fopen(filename_tmp,'wb');
        fwrite(fid,ch,'uint8');
        fclose(fid);
        
        fid=fopen(filename_tmp,'rb');        
        adcdata=fread(fid,[PackPoint*16,FilePackNum],'int32');
        fclose(fid);
        delete(filename_tmp);
        
        data16 = reshape(adcdata,16,PackPoint*FilePackNum);
        %save(savefile,'data16');
        idx=(data16>=2^23);
        data16(idx)=data16(idx)-2^24;
        data16 = data16';
        data=[data data16];
        
        dataidx=udpdata(1,1:FilePackNum)/262144-1024;
        dataidx=dataidx';
        dataidx2=[dataidx2 dataidx];

    end

    %% 


    [mm,nn] = size(data);

    data = data/2^23*2.5/gain;
    
    
    save(savefile,'data');
    
    if mode == 3
        ff = 0:FS/Np:FS/2-FS/Np;
        data = data';
        Xk = abs(fft(data(:,1:Np)'));
        xkkk = fft(data(:,1:Np));
        Xk = Xk';
        Xk = Xk / (Np/2);
        Xk = Xk.^2;
        figure;
        plot(data(CH,:),'.-');
        figure;
        plot(ff,10*log10(Xk(CH,1:Np/2)));
        %figure;
        %specgram(data(:,1),Np,FS,Np,0);
    end
    
    
    
end
    