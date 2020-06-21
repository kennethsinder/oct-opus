% clean environment
clear
clc

% source and destination
datestamp = input('please provide datestamp for /private/fydp1/oct-opus-data/??? : ', 's');
src = strcat('/private/fydp1/oct-opus-data/', datestamp, '/');
if not(exist(src, 'dir'))
    disp(['directory', src, ' does not exist ... exiting'])
    exit
end
dst = strcat('/private/fydp1/enface-images/', datestamp, '/');
mkdir([dst]);

% number of images
N = length(dir([src, '*.png']));
disp([num2str(N), ' images found']);

% load z-axis cross-sections into matlab to build a "cube"
for i = 1:N
    SingleCrossSection = im2double(imread([src, int2str(i), '.png']));
    Scans(:,:,i) = SingleCrossSection;
end
disp('Step 1/3 Complete');

% re-slice the "cube" along the y-axis with squeeze and resize
for i = 1:256
    EnFace = imresize(squeeze(Scans(i,:,:)), [1000 1000]);
    imwrite(EnFace, [dst, int2str(i), '.png']);
end
disp('Step 2/3 Complete');

% the depth you want to have the max proj image over it
% from experiments, it was found that for our images, layers 30-55 provide good results
count = 1;
for i = 30:55
    Im = im2double(imread([dst, int2str(i), '.png']));
    Im = Im./max(max(Im)); % this is for normalizing the image
    Layers(:,:,count) = Im;
    count = count + 1;
end
disp('Step 3/3 Complete');

S = imresize(sum(Layers, 3), [1000 1000]);
m = imresize(max(Layers, [], 3), [1000 1000]);
figure,imshow(m./max(max(m)), [])
figure,imshow(S, [])
