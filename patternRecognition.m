%Kostantinos Mpanakakis AEM:979
clear %clear workspace
clc   %clear command Window
%meros A
A1= rand (2,400); % uniformly distributed random number in the interval (0,1) 
A1(1,:)=A1(1,:)+ randi([2,7],1,400); %+ random number [2,7]
A1(2,:)=A1(2,:)+ ones(1,400);% + a vector of ones

A2= rand (2,100);% uniformly distributed random number in the interval (0,1) 
A2(1,:)=A2(1,:)+ randi([11,14],1,100)./2;%+ a random number [5.5,7]
A2(2,:)=A2(2,:)+ randi([5,10],1,100)./2; %+ a random number [2.5,6]

%plotiing the 500 points
for i=1:400
    plot(A1(1,i),A1(2,i),'g*');
    hold on
    if (i<101) 
        plot(A2(1,i),A2(2,i),'r*');
        hold on
    end
end
title('Pattern Recognition Project Stage A')%title of the graph
legend('ù1','ù2','Location','southwest')%the memorandum of the graph
axis([0 9 0 7])%setting the shown axis length
xlabel('x2')%x-axis label
ylabel('x1')%y-axis label

pause; %w8 for keyboard input to continue to B
%Central Tendency for each class
m1=sum(A1,2)./400;
m2=sum(A2,2)./100;

S1=A1-repmat(m1,1,400);
sigma1=(S1*S1')./(399);

S2=A2-repmat(m2,1,100);
sigma2=(S2*S2')./(99);

%B2 euclid distance
title('Pattern Recognition Project Stage B2"Euclid Distance"')%title of the graph
Euclid_Distance=zeros(2,500);
for i=1:400
    Euclid_Distance(1,i)=sqrt(S1(1,i)^2+S1(2,i)^2);
    Euclid_Distance(2,i)=sqrt((A1(1,i)-m2(1,1))^2+(A1(2,i)-m2(2,1))^2);
end
for i=1:100
    Euclid_Distance(1,i+400)=sqrt((A2(1,i)-m1(1,1))^2+(A2(2,i)-m1(2,1))^2);
    Euclid_Distance(2,i+400)=sqrt(S2(1,i)^2+S2(2,i)^2);
end
Euclid_errors=0;
for i=1:500
    if(i<=400 && Euclid_Distance(1,i)>Euclid_Distance(2,i))
        plot(A1(1,i),A1(2,i),'bs');
        hold on
        Euclid_errors=Euclid_errors+1;
    end
    if(i>400 && Euclid_Distance(1,i)<Euclid_Distance(2,i))
        plot(A2(1,i-400),A2(2,i-400),'bs');
        hold on
        Euclid_errors=Euclid_errors+1;
    end
end
temptext=['Euclid errors: ',num2str(Euclid_errors)];
disp(temptext);
pause;
%B3 Mahalanobis disrance
title('Pattern Recognition Project Stage B3 "Mahalanobis"')%title of the graph
Mahalanobis_Distance=zeros(2,500);
Mahalanobis_S=[S1,S2];%koinos pinakas me tis apostaseis kathe simeiou apo tin klasi pou anoikei
Mahalanobis_sigma=(Mahalanobis_S*Mahalanobis_S')./(499);%the Variance of the 500 points
Mahalanobis_sigma=inv(Mahalanobis_sigma);
Mahalanobis_errors=0;
for i=1:400
    dis1=A1(:,i)-m1(:,1);
    dis2=A1(:,i)-m2(:,1);
    if (sqrt(dis1'*Mahalanobis_sigma*dis1)>sqrt(dis2'*Mahalanobis_sigma*dis2))
        plot(A1(1,i),A1(2,i),'ko');
        hold on
        Mahalanobis_errors=Mahalanobis_errors+1;
    end
end
for i=1:100
    dis1=A2(:,i)-m1(:,1);
    dis2=A2(:,i)-m2(:,1);
    if (sqrt(dis1'*Mahalanobis_sigma*dis1)<sqrt(dis2'*Mahalanobis_sigma*dis2))
        plot(A2(1,i),A2(2,i),'ko');
        hold on
        Mahalanobis_errors=Mahalanobis_errors+1;
    end
end
temptext=['Mahalabobis errors: ',num2str(Mahalanobis_errors)];
disp(temptext);

pause;
%B4 Bayesian Classifier
title('Pattern Recognition Project Stage B4 "Bayesian Classifier"')%title of the graph
C1=-1*(log(2*pi)+(1/2)*log(abs(sigma1)));
C2=-1*(log(2*pi)+(1/2)*log(abs(sigma2)));
Bayesian_errors=0;
for i=1:400
    dis1=A1(:,i)-m1(:,1);
    g1=-(1/2)*(dis1'*inv(sigma1)*dis1) +log(4/5)+C1;
    dis2=A1(:,i)-m2(:,1);
    g2=-(1/2)*(dis2'*inv(sigma2)*dis2) +log(1/5)+C2;
    if g1-g2<0
        plot(A1(1,i),A1(2,i),'mo');
        hold on
        Bayesian_errors=Bayesian_errors+1;
    end
end

for i=1:100
    dis1=A2(:,i)-m1(:,1);
    g1=-(1/2)*(dis1'*inv(sigma1)*dis1) +log(4/5)+C1;
    dis2=A2(:,i)-m2(:,1);
    g2=-(1/2)*(dis2'*inv(sigma2)*dis2) +log(1/5)+C2;
    if g2-g1<0
        plot(A2(1,i),A2(2,i),'mo');
        hold on
        Bayesian_errors=Bayesian_errors+1;
    end
end
    
temptext=['Bayesian errors: ',num2str(Bayesian_errors)];
disp(temptext);
pause
%close
hold off

%C1 PCA
All_data=[A1,A2];
Rx=(1/500)*All_data*All_data';
%syms l
%Lamda=vpasolve((l-Rx(1,1))*(l-Rx(2,2))-Rx(1,2)*Rx(2,1)==0,l)
%Max_Lamda=Lamda(1,1);
%if(Max_Lamda<Lamda(2,1))
 %   Max_Lamda=Lamda(2,1);
%end

[eigenvector,Lamda]=eigs(Rx,1);
eigenvector=abs(eigenvector);
PCAA1=eigenvector'*A1;
PCAA2=eigenvector'*A2;
%plotiing the 500 points
for i=1:400
    plot(PCAA1(1,i),PCAA1(1,i),'g*');
    hold on
    if (i<101) 
        plot(PCAA2(1,i),PCAA2(1,i),'r*');
        hold on
    end
end
title('Pattern Recognition Project Stage C1 "PCA"')%title of the graph
legend('ù1','ù2','Location','southwest')%the memorandum of the graph
%axis([0 10 0 10])%setting the shown axis length
pause
%C2 Euclid PCA
title('Pattern Recognition Project Stage C2 "Euclid PCA"')%title of the graph
PCAm1=sum(PCAA1,2)./400;
PCAm2=sum(PCAA2,2)./100;

PCAS1(1,:)=PCAA1-repmat(PCAm1,1,400);
PCAS1(2,:)=PCAA1-repmat(PCAm2,1,400);

PCAS2(1,:)=PCAA2-repmat(PCAm1,1,100);
PCAS2(2,:)=PCAA2-repmat(PCAm2,1,100);

PCA_Euclid_errors=0;
for i=1:500
    if(i<=400 && abs(PCAS1(1,i))>abs(PCAS1(2,i)))
        plot(PCAA1(1,i),PCAA1(1,i),'bs');
        hold on
        PCA_Euclid_errors=PCA_Euclid_errors+1;
    end
    if(i>400 && abs(PCAS2(1,i-400))<abs(PCAS2(2,i-400)))
        plot(PCAA2(1,i-400), PCAA2(1,i-400),'bs');
        hold on
        PCA_Euclid_errors=PCA_Euclid_errors+1;
    end
end
temptext=['PCA Euclid errors: ',num2str(PCA_Euclid_errors)];
disp(temptext);
hold off
pause
%C3 LDA
title('Pattern Recognition Project Stage C3 "LDA"')%title of the graph
%LDA_g=(m1-m2)'*Mahalanobis_sigma*(A1(:,1)-(1/2)*(m1+m2))-log(1/4)
%A1(:,1)
m0=(4/5)*m1+(1/5)*m2;
Sb=(4/5)*(m1-m0)*(m1-m0)'+(1/5)*(m2-m0)*(m2-m0)';
Sw=(1/2)*(sigma1+sigma2);
LDA_vector=inv(Sw)*Sb; %dikimasa na parw to idioduanisma tou pinaka gia na 
%exw monodiastata dedomena alla den 3erw an einai swsto tha parathesw ton 
%kwdika se sxolia.
for i=1:400
    LDAA1(:,i)=LDA_vector*A1(:,i);
end
for i=1:100
    LDAA2(:,i)=LDA_vector*A2(:,i);
end
%plotiing the 500 points
for i=1:400
    plot(LDAA1(1,i),LDAA1(2,i),'g*');
    hold on
    if (i<101) 
        plot(LDAA2(1,i),LDAA2(2,i),'r*');
        hold on
    end
end
title('Pattern Recognition Project Stage C1 "LDA"')%title of the graph
legend('ù1','ù2','Location','southwest')%the memorandum of the graph
axis([0 10 0 30])%setting the shown axis length
pause

%C4 Euclid LDA
title('Pattern Recognition Project Stage C2 "Euclid LDA"')%title of the graph
%Central Tendency for each class
LDAm1=sum(LDAA1,2)./400;
LDAm2=sum(LDAA2,2)./100;

LDAS1=LDAA1-repmat(LDAm1,1,400);
LDAS12=LDAA1-repmat(LDAm2,1,400);

LDAS2=LDAA2-repmat(LDAm2,1,100);
LDAS21=LDAA2-repmat(LDAm1,1,100);

LDA_Euclid_errors=0;
for i=1:500
    if(i<=400 && sqrt(LDAS1(1,i)^2+LDAS1(2,i)^2)>sqrt(LDAS12(1,i)^2+LDAS12(2,i)^2))
        plot(LDAA1(1,i),LDAA1(2,i),'bs');
        hold on
        LDA_Euclid_errors=LDA_Euclid_errors+1;
    end
    if(i>400 && sqrt(LDAS2(1,i-400)^2+LDAS2(2,i-400)^2)>sqrt(LDAS21(1,i-400)^2+LDAS21(2,i-400)^2))
         plot(LDAA2(1,i-400),LDAA2(2,i-400),'bs');
        hold on
        LDA_Euclid_errors=LDA_Euclid_errors+1;
    end
end
temptext=['LDA Euclid errors: ',num2str(LDA_Euclid_errors)];
disp(temptext);
hold off

% exei arketi diafora kai sta pososta lathous. ithela apla na kanw
%
% title('Pattern Recognition Project Stage C3 "LDA"')%title of the graph
% m0=(4/5)*m1+(1/5)*m2;
% Sb=(4/5)*(m1-m0)*(m1-m0)'+(1/5)*(m2-m0)*(m2-m0)';
% Sw=(1/2)*(sigma1+sigma2);
% LDA_vector=inv(Sw)*Sb;
% 
% 
% [LDA_eigenvector,Lamda]=eigs(LDA_vector,1);
% LDA_eigenvector=abs(LDA_eigenvector);
% LDAA1=LDA_eigenvector'*A1;
% LDAA2=LDA_eigenvector'*A2;
% %plotiing the 500 points
% for i=1:400
%     plot(LDAA1(1,i),LDAA1(1,i),'g*');
%     hold on
%     if (i<101) 
%         plot(LDAA2(1,i),LDAA2(1,i),'r*');
%         hold on
%     end
% end
% title('Pattern Recognition Project Stage C1 "LDA"')%title of the graph
% legend('ù1','ù2','Location','southwest')%the memorandum of the graph
% %axis([0 10 0 10])%setting the shown axis length
% pause
% %C2 Euclid PCA
% title('Pattern Recognition Project Stage C2 "Euclid LDA"')%title of the graph
% LDAm1=sum(LDAA1,2)./400;
% LDAm2=sum(LDAA2,2)./100;
% 
% LDAS1(1,:)=LDAA1-repmat(LDAm1,1,400);
% LDAS1(2,:)=LDAA1-repmat(LDAm2,1,400);
% 
% LDAS2(1,:)=LDAA2-repmat(LDAm1,1,100);
% LDAS2(2,:)=LDAA2-repmat(LDAm2,1,100);
%  
% LDA_Euclid_errors=0;
% for i=1:500
%     if(i<=400 && abs(LDAS1(1,i))>abs(LDAS1(2,i)))
%         plot(LDAA1(1,i),LDAA1(1,i),'bs');
%         hold on
%         LDA_Euclid_errors=LDA_Euclid_errors+1;
%     end
%     if(i>400 && abs(LDAS2(1,i-400))<abs(LDAS2(2,i-400)))
%         plot(LDAA2(1,i-400), LDAA2(1,i-400),'bs');
%         hold on
%         LDA_Euclid_errors=LDA_Euclid_errors+1;
%     end
% end
% temptext=['LDA Euclid errors: ',num2str(LDA_Euclid_errors)];
% disp(temptext);
% hold off
pause %meros D
%plotiing the 500 points
for i=1:400
    plot(A1(1,i),A1(2,i),'g*');
    hold on
    if (i<101) 
        plot(A2(1,i),A2(2,i),'r*');
        hold on
    end
end
title('Pattern Recognition Project Stage D')%title of the graph
legend('ù1','ù2','Location','southwest')%the memorandum of the graph
axis([0 9 0 7])%setting the shown axis length
xlabel('x2')%x-axis label
ylabel('x1')%y-axis label

X_LSE=[A1';A2'];
X_LSE(:,3)=ones(500,1);
y_LSE=[ones(400,1);-1*ones(100,1)];
W_LSE=inv(X_LSE'*X_LSE)*X_LSE'*y_LSE;
x1=0:1:10;
x2=-1*((W_LSE(1,1)/W_LSE(2,1))*x1+(W_LSE(3,1)/W_LSE(2,1)));
plot(x1,x2);
hold on
title('Pattern Recognition Project Stage D1 LSE')%title of the graph
pause
LSE_errors=0;
for i=1:400
    if (X_LSE(i,:)*W_LSE<0)
        plot(X_LSE(i,1),X_LSE(i,2),'md');
        LSE_errors=LSE_errors+1;        
    end
end
for i=401:500
    if (X_LSE(i,:)*W_LSE>0)
        plot(X_LSE(i,1),X_LSE(i,2),'md');
        LSE_errors=LSE_errors+1;
    end
end
temptext=['LSE errors: ',num2str(LSE_errors)];
disp(temptext);
pause
title('Pattern Recognition Project Stage D2 Perceptron')%title of the graph

W_0=[2;-1;0];
x1=0:1:10;
x2=-1*((W_0(1,1)/W_0(2,1))*x1+(W_0(3,1)/W_0(2,1)));
%x2=W_0(2,1);
plot(x1,x2);
hold on
pause
%upologise poia einai lanthasmena
Y_perceptron=[];
for i=1:400
    if (X_LSE(i,:)*W_0<0)
       Y_perceptron=[Y_perceptron;i];
    end
end
for i=401:500
    if (X_LSE(i,:)*W_0>0)
        Y_perceptron=[Y_perceptron;i];
    end
end
m=size(Y_perceptron,1);
m=randi(m);
W_old=W_0;
steps=0;
while (size(Y_perceptron,1)>0 )%&& steps<101)
    m=size(Y_perceptron,1);
    m=randi(m);
    steps=steps+1;
    if y_LSE(Y_perceptron(m))==1
        W_new=W_old+(0.3)*X_LSE(Y_perceptron(m),:)';
    else 
        W_new=W_old-(0.13)*X_LSE(Y_perceptron(m),:)';
    end
    Y_perceptron=[];
    for i=1:400
        if (X_LSE(i,:)*W_new<0)
           Y_perceptron=[Y_perceptron;i];
        end
    end
    for i=401:500
        if (X_LSE(i,:)*W_new>0)
            Y_perceptron=[Y_perceptron;i];
        end
    end
    W_old=W_new;
end

temptext=['Perseptron steps: ',num2str(steps)];
disp(temptext);

x1=0:1:10;
x2=-1*((W_new(1,1)/W_new(2,1))*x1+(W_new(3,1)/W_new(2,1)));
plot(x1,x2);
hold off
pause
%perceptron Batch
%plotiing the 500 points
for i=1:400
    plot(A1(1,i),A1(2,i),'g*');
    hold on
    if (i<101) 
        plot(A2(1,i),A2(2,i),'r*');
        hold on
    end
end
title('Pattern Recognition Project Perceptron Batch')%title of the graph
legend('ù1','ù2','Location','southwest')%the memorandum of the graph
axis([0 9 0 7])%setting the shown axis length
xlabel('x2')%x-axis label
ylabel('x1')%y-axis label
W_0=[2;-1;0];
x1=0:1:10;
x2=-1*((W_0(1,1)/W_0(2,1))*x1+(W_0(3,1)/W_0(2,1)));
%x2=W_0(2,1);
plot(x1,x2);
hold on
pause

%upologise poia einai lanthasmena
Y_perceptron=[];
for i=1:400
    if (X_LSE(i,:)*W_0<0)
       Y_perceptron=[Y_perceptron;i];
    end
end
for i=401:500
    if (X_LSE(i,:)*W_0>0)
        Y_perceptron=[Y_perceptron;i];
    end
end
m=size(Y_perceptron,1);
W_old=W_0;
steps=0;
while (size(Y_perceptron,1)>0 )%&& steps<101)
    m=size(Y_perceptron,1);
    m=randi(m);
    steps=steps+1;
    W_new=W_old;
    for i=1:m
        if y_LSE(Y_perceptron(m))==1
            W_new=W_new+(0.2)*X_LSE(Y_perceptron(m),:)';
        else 
            W_new=W_new-(0.05)*X_LSE(Y_perceptron(m),:)';
        end
    end
    Y_perceptron=[];
    for i=1:400
        if (X_LSE(i,:)*W_new<0)
           Y_perceptron=[Y_perceptron;i];
        end
    end
    for i=401:500
        if (X_LSE(i,:)*W_new>0)
            Y_perceptron=[Y_perceptron;i];
        end
    end
    W_old=W_new;
end

temptext=['Perseptron steps: ',num2str(steps)];
disp(temptext);

x1=0:1:10;
x2=-1*((W_new(1,1)/W_new(2,1))*x1+(W_new(3,1)/W_new(2,1)));
plot(x1,x2);
hold off

pause
close
