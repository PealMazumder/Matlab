### **<u>Histogram Analysis</u>**

```matlab
I = imread('plant.jpg'); % Reading Image and store it to ‘I’
G = rgb2gray(I); % Converting RGB image 'I' to GrayScale Image 'G'
[n, m] = size(G); % n store row number and m store column number
subplot(1, 2 , 1), imshow(G), xlabel('Gray Image');
subplot(1, 2, 2), imhist(G), title('Image Histogram') % imhist is a built-in function to create a histogram

% Simple loop implementation of histogram analysis

H = zeros(1, 256); % Create 1x256 matrix(basically an array) H which is filled by zeros
for i = 1 : n
    for j = 1 : m
        for k = 0 : 255
            if G(i, j) == k % if current pixel gray level value is k then we have to increase it's frequency by 1;
                H(k+1) = H(k + 1) + 1;% (k+1) because matlab uses 1 based indexing
            end
        end
    end
end
figure, plot(H), title('Image Histogram without imhist');
```

****

**<u>Histogram Sliding</u>**

```matlab
I = imread('plant.jpg'); % Reading Image and store it to I
G = rgb2gray(I); % Converting RGB image 'I' to GrayScale Image 'G'
% In Histogram Sliding we can make an image either darker or lighter
bright = G + 80; % shifted to the brigher region
dark = G - 80; % shifted to the darker region

subplot(3, 2, 1), imshow(G), xlabel('Gray Image');
subplot(3, 2, 2), imhist(G), title('Histogram of Gray Image');
subplot(3, 2, 3), imshow(bright), xlabel('Brigher Image');
subplot(3, 2, 4), imhist(bright), title('Histogram of Brighter Image');
subplot(3, 2, 5), imshow(dark), xlabel('Darker Image');
subplot(3, 2, 6), imhist(dark), title('Histogram of Darker Image');
```

**<u>Histogram Stretching</u>**

```matlab
I = imread('plantblur.jpg'); % Reading Image and store it to I
G = rgb2gray(I); % Converting RGB image 'I' to GrayScale Image 'G'

Rmin = double(min(G(:))); % Rmin = minimum intensity value present in the whole image
Rmax = double(max(G(:))); % Rmax = maximum intensity value present in the whole image

outImage = zeros(size(G));  % create a matrix of zeros equal to gray image size to store the output image

[n, m] = size(G);  % n store row number and m store column number
for i = 1 : n
    for j = 1 : m
        currIntensity = double(G(i, j)); % current pixel gray level value in double
        outImage(i, j) = ((currIntensity - Rmin)*255)/(Rmax - Rmin); % According to the contrast stretching General Formula
    end
end
outImage = uint8(outImage); % converted 'double' type outImage to 'uint8' type as we kmow gray image pixel value could not be a 'double' type 

%ploting images
subplot(2, 2, 1), imshow(G), xlabel('Gray Image');
subplot(2, 2, 2), imhist(G), title('Histogram of Gray Image');
subplot(2, 2, 3), imshow(outImage), xlabel('Stretched Image');
subplot(2, 2, 4), imhist(outImage), title('Histogram of Stretched image');
```

### **<u>Histogram Equalization</u>**

```matlab
=>Using Built-in Function

I = imread('plant.jpg'); % Reading Image and store it to variable I
G = rgb2gray(I); % Converting RGB image 'I' to GrayScale Image 'G'
Heq = histeq(G); % histeq is a built-in function for Histogram Equalization

%ploting required images
subplot(2, 2, 1); imshow(I), xlabel('Original Image');
subplot(2, 2, 2); imshow(G), xlabel('Gray Image');
subplot(2, 2, 3); imhist(G), title('Histogram of the Image');
subplot(2, 2, 4); imshow(Heq), xlabel('Histogram Equalization');

```

```matlab
=> Without Built-in Function
I = imread('plant.jpg'); % Reading Image and store it to I
G = rgb2gray(I); % Converting RGB image 'I' to GrayScale Image 'G'
[n, m] = size(G); % n store row number and m store column number

G = double(G); % Converting image gray level values to double for calculation purpose

H1 = zeros(1, 256); % Create 1x256 matrix(basically an array) H1 which is filled by zeros
for i = 1 : n
    for j = 1 : m
        for k = 0 : 255
            if G(i, j) == k % if current pixel gray level value is k then we have to increase it's frequency by 1;
                H1(k+1) = H1(k + 1) + 1; % (k+1) because matlab uses 1 based indexing
            end
        end
    end
end

pmf = ((1* hist1)/(sg(1) * sg(2))); % pmf stands for Probability Mass Function
                               % pmf = Histogram / Total number of pixel

cdf = zeros(1, 256); % Create 1x256 matrix(basically an array) cdf which is filled by zeros
                     % cdf stands for Cumulative Distribution Function
cdf(1) = pmf(1); % for cdf calculation first we have to copy the first element as it is.
for i = 2 : 256
    cdf(i) = cdf(i-1) + pmf(i); %It's kinda Prefix sum calculation of pmf
end

cdf = round(255*cdf); % transforming cdf values to [0, 255] range

eqImage = zeros(size(G)); % create a matrix of zeros equal to gray image size to store the equalized image
for i = 1 : n
    for j = 1 : m
        id = (G(i, j) + 1); % matlab indexing is 1 based that's why we need to add 1
        eqImage(i, j) = cdf(id);
    end
end

%Calculating equalized image histogram
H2 = zeros(1, 256);
for i = 1 : n 
    for j = 1 : m
        for k = 0 : 255
            if eqImage(i, j) == k  % if current pixel gray level value is k then we have to increase it's frequency by 1;
                H2(k+1) = H2(k + 1) + 1; % (k+1) because matlab uses 1 based indexing
            end
        end
    end
end

%ploting images
%As we converted our images to double values so to plot this images, again
%we need to convert this to uint8
subplot(2, 3, 1), imshow(uint8(G)), xlabel('Gray Image');
subplot(2, 3, 2), plot(hist1), title('Histogram of Gray Image');
subplot(2, 3, 3), imshow(uint8(eqImage)), xlabel('Equalized Image');
subplot(2, 3, 4), plot(hist2), title('Histogram of Equalized Image');

```

**<u>Median Filter</u>**

```matlab
I = imread('plant.jpg'); % Reading Image and store it to 'I'
NoisyImage = imnoise(I, 'salt & pepper', 0.4); %Adding salt & pepper nosie and density of the noise is 0.4

%splitting the channels(RGB) of the image
redch = NoisyImage(:, :, 1);
greench = NoisyImage(:, :, 2);
bluech = NoisyImage(:, :, 3);

% applying median filter on all channels
redch = medfilt2(redch, [3 3]); % size of the filter is 3x3
greench = medfilt2(greench, [3 3]);
bluech = medfilt2(bluech, [3, 3]);

%Concatenating three channels to reform the RGB image
final_image = cat(3, redch, greench, bluech);% 3 for 3 dimensional array

%ploting required images
subplot(1, 2, 1), imshow(NoisyImage), xlabel('Noisy Image');
subplot(1, 2, 2), imshow(final_image), xlabel('Image after noise removal');

```

**<u>Gaussian Filter</u>**

```matlab
I = imread('plant.jpg'); % Reading Image and store it to 'I'
NoisyImage = imnoise(I, 'salt & pepper', 0.04); %Adding salt & pepper nosie and density of the noise is 0.04

%splitting the channels(RGB) of the image
redch = NoisyImage(:, :, 1);
greench = NoisyImage(:, :, 2);
bluech = NoisyImage(:, :, 3);

%defining the filter
filter = fspecial('gaussian', [10 10], 4);% size of the filter is 10x10
                                          % standared deviation value is 4


% applying filter on all channels
redch = imfilter(redch, filter); 
greench = imfilter(greench, filter);
bluech = imfilter(bluech, filter);

%Concatenating three channels to reform the RGB image
final_image = cat(3, redch, greench, bluech);% 3 for 3 dimensional array

%ploting required images
subplot(1, 2, 1), imshow(NoisyImage), xlabel('Noisy Image');
subplot(1, 2, 2), imshow(final_image), xlabel('Image after noise removal');

```

**<u>Wavelet Transformation Analysis</u>**

```matlab
I = imread('plant.jpg');% Reading Image and store it to I
G = rgb2gray(I); % Converting RGB image 'I' to GrayScale Image 'G'
G = double(G);% Converting image gray level values to double for calculation purpose

n = 2; % level of DWT decomposition

dwtmode('per'); %Image extension Mode

[C, S] = wavedec2(G, n, 'haar');% 1 level DWT
[cHn, cVn, cDn] = detcoef2('all', C, S, n); % Extracting Detailed Coeff. at level n
cAn = appcoef2(C, S, 'haar', n);% Extracting Approximation Coeff. at level n

subplot(2, 2, 1), imshow(uint8(cAn)), title(strcat('Approximation Coefficients at level ', num2str(n)));
subplot(2, 2, 2), imshow(uint8(cHn), []), title(strcat('Horizontal Detailed Coefficients at level ', num2str(n)));
subplot(2, 2, 3), imshow(uint8(cVn), []), title(strcat('Vertical Detailed Coefficients at level ', num2str(n)));
subplot(2, 2, 4), imshow(uint8(cDn), []), title(strcat('Diagonal Detailed Coefficients at level ', num2str(n)));

imgr = waverec2(C, S, 'haar');
figure(2), imshow(uint8(imgr)), title('Reconstructed Image');

```

