clc
close all
clear all
%% Prepare constants and Compute the fitting curve
% edit the path for the image folder
path= 'E:\2016-12-05\Images5\aligned_cog\';
mkdir 'E:\2016-12-05\Images5\aligned_cog\flatten_Image\'

fit_type = 'poly1'; % or 'poly2'
shift_pixel = 20; % define if the flatten image needs to be shifted downwards by 

% count number of images in the folder, for for loop`
folder_info =  struct2cell(dir([path '/*.png']));
img_fnames = folder_info(1,:);
num_imgs = size(img_fnames,2);
% read the first image to see the pixel and ascan number
[pixel,ascan] =size(imread([path,char(img_fnames(1))]));
% cut in a scan, starting from pixle cut_a
cut_a = 80;
% cut in b scan direction, symmetrical cut, 
% left 20 pixel and right 20 pixel
cut_b = 20;
    
%{ for testing purpose, uncomment and select the image/s you want to flatten
% start_ind = 1250;
% num_imgs = 1250;
   
% standard deviation for creating gaussian filter
sigma = 1; 

% threshold for taking the binary image
Th=100;

for i=start_ind:num_imgs
    b = imread([path,char(img_fnames(i))]);
    b = b(:,cut_b+1:ascan-cut_b) ;

 %% Smooth out the selected region 
    G = fspecial('gaussian',3,sigma); % Caussian kernel
    bf=  imfilter(b,G,'same');  % smooth image by Gaussiin convolution
    bfc = bf(cut_a:end,:);
    BW = bfc>=Th; % computing the binary image
    
    % y size of the fitting curve. should be the num of A scans for cutted
    % image
    ys=size(bf,2);
    yy = zeros(1,ys);
    for j=2:ys
        xBWn=find(BW(:,j)==1);
        if isempty(xBWn)
            yy(j)=yy(j-1);
        else
            yy(j)=xBWn(1);
        end         
    end
    shift_sig=shift_pixel-yy;
    
    % generate x axis for fitting function
    x = linspace(1,ys,ys);
    
    f = fit(x',shift_sig',fit_type,'Robust','Bisquare');
    switch fit_type
        case 'poly1'
            cvals = coeffvalues(f);
            a0 = cvals(2);
            a1 = cvals(1); 
            yy1 =  a1*x + a0;

        case 'poly2'
            a0 = cvals(3);
            a1 = cvals(2); 
            a2 = cvals(1);
            yy1 = a2*x.^2 + a1*x + a0;
    end
    
    yy1 = round(yy1);
    
    %% shift the image from the fitting curve
    imm = uint8(zeros(size(b)));
    immf = uint8(zeros(size(b)));
for ind =1:ys
    sig=b(:,ind);
    sigf=bf(:,ind);
    Imm(:,ind)=circshift(sig,yy1(ind)); 
    Immf(:,ind)=circshift(sigf,yy1(ind)); 
end

%% plot three images and see the result for testing purpose
% comment out if automated for the looping
subplot(1,3,1) , imshow(b,[0 255]);
title('Original Cutted Image');
 set(gca, 'Position', [0.01, 0,0.32,1]);
subplot(1,3,2) , imshow(Imm,[0 255]);
title('Flattened Cutted Image')
 set(gca, 'Position', [0.34, 0, 0.32, 1]);
 subplot(1,3,3) , imshow(Immf,[0 255]);
title('Flattened Filtered Cutted Image')
 set(gca, 'Position', [0.67, 0, 0.32, 1]);
 
 %% uncomment below to save the images
% imwrite(Immf,[path,'flatten_Image\',int2str(i),'.png']);
% imwrite(b,sprintf('im.png'));

end

