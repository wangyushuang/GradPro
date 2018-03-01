%% ˫Բ�����Ȼָ���� 
%����Sensor Array Analyzer�������ɴ��룬�޸Ĳ���
%double circle array analyze 
%���������ָ���䷽λ�ǽ����-3dB������Χ
clear;close all;
R1=0.025;
R2=0.05;
N1=6;
N2=6;
f=50000;%frequency
c=1500;%velocity
%Element Position
x=[R1*cos([0:1:N1-1]*pi/3),R2*cos([0:1:N2-1]*pi/3)];%��Ԫx����
y=[R1*sin([0:1:N1-1]*pi/3),R2*sin([0:1:N2-1]*pi/3)];%��Ԫy����
z=zeros(1,N1+N2);%��Ԫz����
ep=[x;y;z]; %��Ԫλ�þ���
%Elemet Normal
en=zeros(2,N1+N2);
%���Ʋ����γɵĽǶ�
theta_s=0; %��Χ��0-180deg
phy_s=75; %��Χ��0-90deg,90deg��׼��������
SA = [theta_s;phy_s];

%% 3D
% Create an arbitrary geometry array
h = phased.ConformalArray();

 
h.ElementPosition =ep;
h.ElementNormal =en;
wind = ones(1,N1+N2);
h.Taper = wind;
%Create Isotropic Antenna Element
el = phased.IsotropicAntennaElement;
h.Element = el;
%Calculate Steering Weights
w = zeros(getNumElements(h), length(f));
SV = phased.SteeringVector('SensorArray',h, 'PropagationSpeed', c);
%Find the weights
for idx = 1:length(f)
    w(:, idx) = step(SV, f(idx), SA(:, idx));
end
%Plot 3d graph
fmt = 'polar';
[pat,az,el]=pattern(h,f,'PropagationSpeed', c, 'Type','directivity', ...
    'CoordinateSystem', fmt);
%Create figure, panel, and axes
fig = figure;
panel = uipanel('Parent',fig);
hAxes = axes('Parent',panel,'Color','none');

pattern(h,f,'PropagationSpeed', c, 'Type','directivity', ...
    'CoordinateSystem', fmt,'weights', w(:,1));

%Adjust the view angles
view(hAxes,[135 20]);
title = get(hAxes, 'title');
title_str = get(title, 'String');
%Modify the title
[Fval, ~, Fletter] = engunits(f);
title_str = [title_str sprintf('\n') num2str(Fval) ' ' Fletter 'Hz '  ...
    ' steered at ' num2str(SA(1)) '��AZ ' num2str(SA(2)) '��EL'];
set(title, 'String', title_str);

%% 2D ��λ�ǽ����棺�ⲿ�ֶ���
cutAngle = SA(2);%�ڸø����Ƿ��������
%Assign number of phase shift quantization bits
PSB = 0;
%Create figure, panel, and axes
fig = figure;
panel = uipanel('Parent',fig);
hAxes = axes('Parent',panel,'Color','none');
NumCurves = length(f);
%Calculate Steering Weights
w = zeros(getDOF(h), NumCurves);
for idx = 1:length(f)
    SV = phased.SteeringVector('SensorArray',h, 'PropagationSpeed', c, ...
        'NumPhaseShifterBits', PSB(idx));
    w(:, idx) = step(SV, f(idx), SA(:, idx));
end
%Plot 2d graph
fmt = 'rectangular';
[pat_a,az,el]=pattern(h, f, -180:0.01:180, cutAngle, 'PropagationSpeed', c, 'Type', ...
    'directivity', 'CoordinateSystem', fmt ,'weights', w);
pattern(h, f, -180:180, cutAngle, 'PropagationSpeed', c, 'Type', ...
    'directivity', 'CoordinateSystem', fmt ,'weights', w);
axis(hAxes,'square')
%Create legend
legend_string = cell(1,NumCurves);
lines = findobj(gca,'Type','line');
for idx = 1:NumCurves
    [Fval, ~, Fletter] = engunits(f(idx));
    if size(SA, 2) == 1
        az_str = num2str(SA(1,1));
        elev_str = num2str(SA(2,1));
    else
        az_str = num2str(SA(1, idx));
        elev_str = num2str(SA(2, idx));
    end
    if PSB(idx)>0
        legend_string{idx} = [num2str(Fval) Fletter 'Hz;' num2str(SA(1, ...
            idx)) 'Az' ' ' elev_str 'El' ';' ...
            num2str(PSB(idx)) '-bit Quantized'];
    else
        legend_string{idx} = [num2str(Fval) Fletter 'Hz;' num2str(SA(1, ...
            idx)) 'Az' ' ' elev_str 'El'];
    end
end
legend(legend_string, 'Location', 'southeast');
hold on 
plot(linspace(-180,180,10000),ones(1,10000)*(max(pat_a)-3),'r') 
xlim([-180 180])

pos_az=[];
err=1e-2;
for k=2:length(pat_a)-1%round((length(pat_a)+1)/4):round((3*length(pat_a)+3)/4) %��90��
    if (pat_a(k)-max(pat_a)+3<err)&&((pat_a(k-1)-max(pat_a)+3)*(pat_a(k+1)-max(pat_a)+3)<0)
        pos_az=[pos_az k];
    end
end

% plot(az(pos_az),pat_a(pos_az),'or') %���ڷ�λ�ǿ��������壬���������ͼ
if length(pos_az)==4
    wid_AZ=(az(pos_az(3))+az(pos_az(4)))/2-(az(pos_az(1))+az(pos_az(2)))/2;%-3dB������
else
     wid_AZ=[];
end

%% 2D ����������
cutAngle = SA(1);%��λ��
%Assign number of phase shift quantization bits
PSB = 0;
%Create figure, panel, and axes
fig = figure;
panel = uipanel('Parent',fig);
hAxes = axes('Parent',panel,'Color','none');
NumCurves = length(f);
%Calculate Steering Weights
w = zeros(getDOF(h), NumCurves);
for idx = 1:length(f)
    SV = phased.SteeringVector('SensorArray',h, 'PropagationSpeed', c, ...
        'NumPhaseShifterBits', PSB(idx));
    w(:, idx) = step(SV, f(idx), SA(:, idx));
end
%Plot 2d graph
fmt = 'rectangular';
[pat_e,az,el]=pattern(h, f, cutAngle, 0:0.05:90, 'PropagationSpeed', c, 'Type', ...
    'directivity', 'CoordinateSystem', fmt ,'weights', w);
pattern(h, f, cutAngle, 0:90, 'PropagationSpeed', c, 'Type', ...
    'directivity', 'CoordinateSystem', fmt ,'weights', w);
axis(hAxes,'square')
%Create legend
legend_string = cell(1,NumCurves);
lines = findobj(gca,'Type','line');
for idx = 1:NumCurves
    [Fval, ~, Fletter] = engunits(f(idx));
    if size(SA, 2) == 1
        az_str = num2str(SA(1,1));
        elev_str = num2str(SA(2,1));
    else
        az_str = num2str(SA(1, idx));
        elev_str = num2str(SA(2, idx));
    end
    if PSB(idx)>0
        legend_string{idx} = [num2str(Fval) Fletter 'Hz;' num2str(SA(1, ...
            idx)) 'Az' ' ' elev_str 'El' ';' ...
            num2str(PSB(idx)) '-bit Quantized'];
    else
        legend_string{idx} = [num2str(Fval) Fletter 'Hz;' num2str(SA(1, ...
            idx)) 'Az' ' ' elev_str 'El'];
    end
end
legend(legend_string, 'Location', 'southeast');
hold on 
plot(linspace(0,90,10000),ones(1,10000)*(max(pat_e)-3),'r') 

pos_el=[];
err=1e-1;
for k=2:length(pat_e)-1
    if (pat_e(k)-max(pat_e)+3<err)&&((pat_e(k-1)-max(pat_e)+3)*(pat_e(k+1)-max(pat_e)+3)<0)
        pos_el=[pos_el k];
    end
end
plot(el(pos_el),pat_e(pos_el),'or')
xlim([0 90])
if length(pos_el)==4
    wid_EL=(el(pos_el(3))+el(pos_el(4)))/2-(el(pos_el(1))+el(pos_el(2)))/2;%-3dB������
elseif length(pos_el)==2
    wid_EL=180-(el(pos_el(2))+el(pos_el(1)));
else
    wid_EL=[];
end

% string=['Wid_AZ:' num2str(wid_AZ) '�� ' 'Wid_EL:' num2str(wid_EL) '��'];
string=['�������:' num2str(wid_EL) '��'];%����������
disp(string)
%% Sensor Array Analyzerͼ�ν���������
% Element Position:ep
% Element Normal:en
% Signal Frequencies:f
% Propagation Speed:c
