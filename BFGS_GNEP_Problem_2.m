clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  Method selection

    syms armizo WWP
    %method0=armizo
    method0=WWP
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    n=7      %Number of variables

for tttt=1:100            %%%% Total number of random initial points selection
    if tttt<50
    %a=20+rand(1,n)*.1
    a=10+rand(1,n)*.1
    norm(a)
    else
    %a=28+rand(1,n)*.1
    a=20+rand(1,n)*.1
    norm(a)
    end
    
%%%%%%%%%%%  Parameters values
mm=100
rrrr=2
err=10^-8
c=.01
c1=c/3
theta=1-c


beta=10^-8
alpha1=.01
alpha2=3




tic

b=a; 




%%% objective functions


syms xx1 xx2 lm11 lm12 lm21 lm22 mu 
z=[xx1 xx2 lm11 lm12 lm21 lm22 mu]

theta1 =@(x1,x2) -x1^2-(8/3)*x1*x2+34*x1
theta2 =@(x1,x2) -x2^2-(5/4)*x1*x2+24.25*x2

h11(xx1)= -xx1
h12(xx1)=xx1-10

h21(xx2)=-xx2
h22(xx2)=xx2-10
 
shared(xx1,xx2)=xx1+xx2-15
%g2(x1,x2)= x1+x2-1
ps=@(x1,x2) sqrt(x1^2+x2^2)-(x1+x2)
phi(xx1,xx2)=ps(xx1,xx2)^2


kk(xx1,xx2)=phi(xx1,xx2)


phi11(xx1,lm11)= kk(-h11,lm11)                
phi12(xx1,lm12)= kk(-h12,lm12)                

phi21(xx2,lm21)= kk(-h21,lm21)               
phi22(xx2,lm22)= kk(-h22,lm22)               


phi_s(xx1,xx2,mu)= kk(-shared(xx1,xx2),mu)           

 
f1(xx1,xx2,lm11,lm12,mu)= gradient(theta1+ lm11*h11+lm12*h12 + shared(xx1,xx2)*mu,xx1)
f2(xx1,xx2,lm21,lm22,mu)= gradient(theta2+ lm21*h21+lm22*h22 + shared(xx1,xx2)*mu,xx2)
 
f(xx1,xx2,lm11,lm12,lm21,lm22,mu)=[f1(xx1,xx2,lm11,lm12,mu);f2(xx1,xx2,lm21,lm22,mu)]

phi(xx1,xx2,lm11,lm12,lm21,lm22)=[phi11(xx1,lm11);phi12(xx1,lm12);phi21(xx2,lm21);phi22(xx2,lm22)]


F(xx1,xx2,lm11,lm12,lm21,lm22,mu)=[f;phi(xx1,xx2,lm11,lm12,lm21,lm22);phi_s(xx1,xx2,mu)]



% %% Reformulated system


n=length(z)
x=sym('x',[1,n]);


FF=subs(F,z,x)
psi=.5*norm(FF)^2
D0=eye(n)



F=subs(psi,z,x)
f0=subs(FF,x,a)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





F0=vpa(subs(F,x,a));
G=gradient(F,x);
G0=vpa(subs(G,x,a));

D0=eye(n);
counter=1;

%%%%%%  Main algorithm starts

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Stopping criteria norm(G0) < err

  while   norm(G0)>err  %%%%  Stopping condition
     

d=-D0^-1*G0;                  %%% Descent direction



test_for_descent_dir=vpa((transpose(G0)*d));   %%%% Check whether the direction is descent or not

if test_for_descent_dir>=0
    disp('d is not a descent direction')
    break;
end
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Computation of step-length
syms s;
z=a+mm*(rrrr)^-s*d' ;                             % This is our iteration 

F_new(s)=vpa(subs(F,x,z));
G_new(s)=vpa(subs(G,x,z));

s=0;

if method0==armizo

while F_new(s)> F0 + c*mm*rrrr^-s*G0.'*d | F_new(s)< F0 + (1-c)*mm*rrrr^-s*G0.'*d 
    s=s+1;
    if s>200
        disp('Value of step size is higher than 200')
        fprintf('We were in iteration #%d\n',counter)
        break;
    end
end
end

if method0==WWP

while F_new(s)> F0 + c*mm*rrrr^-s*G0'*d +mm*rrrr^-s*min(-c1*G0'*d,(c*mm*rrrr^-s*norm(d)^2)/2) | G_new(s)'*d< theta*G0'*d + min(-c1*G0'*d,(c*mm*rrrr^-s*norm(d)^2))
s=s+1;
    if s>200
        disp('Value of step size is higher than 200')
        fprintf('We were in iteration #%d\n',counter)
        break;
    end
end
end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('value of step size')
sss=mm*rrrr^-s;
disp(sss)

z=a+mm*rrrr^-s*d.';    %%% Updated value of a


    %%%%%%%%%%%%%%%%%%%%%%%      Computation of BFGS update matrix

F0=vpa(subs(F,x,a));
F1=vpa(subs(F,x,z));
G0=subs(G,x,a);
G1=(subs(G,x,z));
y1=(G1-G0);

if F0-F1<0
    disp('<strong>Not a descent direction</strong>')
    break;
end

s1=z'-a';          
kkk=D0;
A0=(6*(F0-F1)+3*(G1+G0)'*s1)/(norm(s1))^2;
Ak=max(A0,0);
yk=y1+Ak*s1;

if vpa(norm(G0))>=1
    alpha=alpha1;
else
    alpha=alpha2;
end


if method0==armizo

D1=piecewise((yk'*s1)/norm(s1)^2>beta*norm(G0)^alpha, vpa(D0+(yk*yk')/(yk'*s1)-(D0*s1*s1'*D0)/(s1'*D0*s1)),D0);

end
if method0==WWP

    D1=vpa(D0+(yk*yk')/(yk'*s1)-(D0*s1*s1'*D0)/(s1'*D0*s1));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Check the eigen values of BFGS matrix

ee=eig(D1);   
 if ee<0
disp('Hessian approximation matrix is not positive definite')

break;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



D0=vpa(D1);

   
quasi=abs(vpa(D1*s1)-vpa(yk));              
if abs(vpa(D1*s1)-vpa(yk))<=err^2
    disp('Quasi equation satisfied')
else
    disp('Quasi Newton equation not satisfied')
    break;
end



  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%    Iteration update
a=z;
F0=(subs(F,x,a));

G0=(subs(G,x,a));
G2=(norm(G0));
m(counter,:)=[counter s sss vpa(norm(F0),4) vpa(G2,4)];
fprintf('Just finished main iteration #%d and sub-iter ##%d ',tttt,counter)
counter=counter+1;
disp('Norm of gradient')
vpa(norm(G0),2)
disp('Norm of function')
vpa(norm(f0),2)
%fprintf('<strong> BFGS by %s \n</strong>',method0)
f0=subs(FF,x,a);
end


syms iter  step_chosen step_size norm_of_fun norm_of_grad;
fprintf('Total number of Iterations : #%d\n',counter-1)
[iter step_chosen step_size norm_of_fun norm_of_grad;m]
disp('initial vaue was: ')
b


disp('Root of the function:')
vpa(a,4)
disp('Norm of  Gradient')
G0=(subs(G,x,a));
G2=vpa(norm(G0),2)
disp('Functional value')
vpa(subs(F,x,a),2)
disp('Function')
f0=vpa(subs(FF,x,a),2)
vpa(norm(f0),2)
fprintf('<strong> BFGS by %s   in iterations %d \n </strong>',method0,counter-1)
t=toc
distance=norm(b)
total_ttt(tttt,:)=vpa([tttt distance t counter-1 G2 a 00000 b],4)
tttt=tttt+1;
clear s
end
total_ttt
tt11=vpa(total_ttt,4)



