%% INITIATE THE CODE
clear
clc

%% Give the Directory
Dir='./2015-09-07Images46/';
mkdir './2015-09-07Images46/En-Face';
% mkdir 'G:\2018-05-26\Images39\Aligned'
%% save 3D complex data matrix with all B-scans

% number of OMAG images
N = length(dir([Dir, 'OMAG Bscans/', '*.png'])) - length(dir([Dir, 'OMAG Bscans/', '*_*.png']));

%% number of images in the OMAG
for i = 1:N
    disp(i)
    iMat = im2double(imread([Dir, 'OMAG Bscans/', int2str(i), '.png']));
    Int(:,:,i) = iMat;
end

%% number of vertical pixels (depth)
for kk = 1:1000
    disp(kk)
    dEnFace = imresize(squeeze(Int(kk,:,:)), [1000 1000]);
    imwrite(dEnFace, [Dir, 'En-Face/', int2str(kk), '.png']);
end
