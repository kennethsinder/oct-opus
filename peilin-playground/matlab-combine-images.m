% clean environment
clear
clc

%% source and destination
src = '/private/fydp1/oct-opus-data/2015-09-07-Images-46/';
dst = '/private/fydp1/enface-images/2015-09-07-Images-46/';
mkdir([dst]);

% number of images
N = length(dir([src, '*.png'])) - length(dir([src, '*_*.png']));

%% load images into matlab
for i = 1:N
    disp(i)
    iMat = im2double(imread([src, int2str(i), '.png']));
    Int(:,:,i) = iMat;
end

%% number of vertical pixels (depth)
for kk = 1:1000
    disp(kk)
    dEnFace = imresize(squeeze(Int(kk,:,:)), [1000 1000]);
    imwrite(dEnFace, [dst, int2str(kk), '.png']);
end

count = 1;

% the depth you want to have the Max_Proj image over it
for i = 680:750
    Im = im2double(imread([dst, int2str(i), '.png']));
    Im = Im./max(max(Im)); % this is for normalizing the image
    Int(:,:,count) = Im;
    count = count + 1;
end

S = imresize(sum(Int,3),[1000 1000]);
m = imresize(max(Int,[],3),[1000 1000]);
figure,imshow(m./max(max(m)),[])
figure,imshow(S,[])
