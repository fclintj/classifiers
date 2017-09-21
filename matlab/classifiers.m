% classasgnl.m
% Sax0 = classasgntrain1.dat(:,1:2);mple classifier program
% Load the training data and divide into the different classes
data = importdata('classasgntrain1.dat');
x0 = data(:,1:2)'; % is data vectors for class O (2 x N0)
N0 = size(x0,2);
x1 = data(:,3:4)'; % data vectors for class 1 (2 x N1)
N1 = size(x1,2);
N = N0 + N1;



%%
% Linear regression classifier
% Build the X matrix
X = [ones(N0,1) x0';
    ones(N1,1) x1'];

% Build the indicator response matrix
Y = [ones(N0,1) zeros(N0,1)
    zeros(N1,1) ones(N1,1)];

% Find the parameter matrix
Bhat = (X'*X) \ X'* Y;

% Find the approximate response
Yhat = X*Bhat;
Yhathard = Yhat > 0.5; % threshold into different classes

num_err = sum(sum(abs(Yhathard - Y)))/2; % count the total number of errors
err_rate_linregress_train = num_err / N;

% Now test on new (testing data)
Ntest0 = 10000; % number of class 0 points to generate Ntest
Ntest1 = 10000;

xtest0 = gendat2(0,Ntest0); % generate the test data for class O
xtest1 = gendat2(0,Ntest0); 
num_err = 0;

for i=1:Ntest0
    yhat = [1 xtest0(:,i)']*Bhat;
    if(yhat(2) > yhat(1)) % error: chose class 1 over class 0
    num_err = num_err+1;
    end
end

for i=1:Ntest1
    yhat = [1 xtest1(:,i)']*Bhat;
    if(yhat(1) > yhat(2)) % error: chose class 0 over class 1
    num_err = num_err+1;
    end
end

err_rate_linregress_test = num_err / (Ntest0 + Ntest1);

% Plot the performance across the window (that is, plot the classification
% regions)
xmin = min([x0(1,:) x1(1,:)]); xmax = max([x0(1,:) x1(1,:)]);
ymin = min([x0(2,:) x1(2,:)]); ymax = max([x0(2,:) x1(2,:)]);

xp1 = linspace(xmin,xmax,100);
yp1 = linspace(ymin,ymax,100);

redpts = []; % class 1 estimates
greenpts = [];% class 0 estimatees

% % plot the data
clf;
plot(xtest0(1,:), xtest0(2,:),'gx');
hold on;
plot(xtest1(1,:), xtest1(2,:),'ro');
xlabel('x_0');
ylabel('x_1');

% loop over all points
for x = xp1
    for y = yp1
        yhat = [1 x y]*Bhat;
        if(yhat(1) > yhat(2)) % choose class 0 over class 1
            greenpts = [greenpts [x;y]];
        else
            redpts = [redpts [x;y]];
        end
    end
end
plot(greenpts(1,:), greenpts(2,:),'g.','MarkerSize',0.25);
plot(redpts(1,:), redpts(2,:),'r.','MarkerSize',0.25);
axis tight