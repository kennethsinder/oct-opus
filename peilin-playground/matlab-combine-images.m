% clean environment
clear
clc

%% source and destination
src = '/private/fydp1/oct-opus-data/2015-09-07-Images-46/';
dst = '/private/fydp1/enface-images/2015-09-07-Images-46/';
mkdir([dst]);

% number of images
N = length(dir([src, '*.png']));

%% load images into matlab
for i = 1:N
    disp(i)
    iMat = im2double(imread([src, int2str(i), '.png']));
    Int(:,:,i) = iMat;
end

%% number of vertical pixels (depth)
for i = 1:256
    disp(i)
    dEnFace = imresize(squeeze(Int(i,:,:)), [1000 1000]);
    imwrite(dEnFace, [dst, int2str(i), '.png']);
end

% the depth you want to have the Max_Proj image over it
count = 1;
for i = 30:55
    Im = im2double(imread([dst, int2str(i), '.png']));
    Im = Im./max(max(Im)); % this is for normalizing the image
    Layers(:,:,count) = Im;
    count = count + 1;
end

S = imresize(sum(Layers, 3), [1000 1000]);
m = imresize(max(Layers, [], 3), [1000 1000]);
figure,imshow(m./max(max(m)), [])
figure,imshow(S, [])
