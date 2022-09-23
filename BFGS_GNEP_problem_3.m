clear
clc


for tttt=1:100  % Number of randomly starting points
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         Method selection
    syms armizo WWP
    %method0=armizo
    method0=WWP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%     Initial values
    if tttt<30
        %a=rand(1,5)*.1
        %a=.75+rand(1,5)*.1
        %a=3+rand(1,5)*.1
         %a=8+rand(1,5)*.1
         a=25+rand(1,5)*.1
        norm(a)
    end

    if tttt<60 & tttt>=30
        %a=.25+rand(1,5)*.1
        %a=1.25+rand(1,5)*.1
        %a=4+rand(1,5)*.1
         %a=16+rand(1,5)*.1
         a=30+rand(1,5)*.1
        norm(a)
    end

    if tttt>=60
        %a=.3+rand(1,5)*.1
        %a=2+rand(1,5)*.1
        %a=6+rand(1,5)*.1
        %a=21+rand(1,5)*.1
        a=35+rand(1,5)*.1
        norm(a)
    end

    
%%%%%%%%%%%  Parameters values
n=5      %Number of variables
mm=100
rrrr=2
err=10^-8
c=.01
c1=c/3
theta=1-c
beta=10^-8
alpha1=.01
alpha2=3






tic % time starts

b=a;
    


%%% objective functions


syms(sym('x',[1 5]))
x=sym('x',[1 5])
syms xx1 xx2 lm1 lm2 mu
z=[xx1,xx2,lm1 lm2 mu]
theta1 =@(x1,x2) .5*x1^2-x1*x2
theta2 =@(x1,x2) x2^2+x1*x2



h1(xx1)= -xx1
h2(xx2)=-xx2
 
s(xx1,xx2)=1-xx1-xx2
%g2(x1,x2)= x1+x2-1

ps=@(x1,x2) sqrt(x1^2+x2^2)-(x1+x2)
phii(xx1,xx2)=ps(xx1,xx2)^2

phi1(xx1,lm1)= phii(-h1,lm1)
phi2(xx2,lm2)= phii(-h2,lm2)


phi_s(xx1,xx2,mu)= phii(-s,mu)

 
f1(xx1,xx2,lm1,mu)= gradient(theta1+ lm1*h1 + s(xx1,xx2)*mu,xx1)
f2(xx1,xx2,lm2,mu)= gradient(theta2+ lm2*h2 + s(xx1,xx2)*mu,xx2)
 
f(xx1,xx2,lm1,lm2,mu)=[f1(xx1,xx2,lm1,mu);f2(xx1,xx2,lm2,mu)]

phi(xx1,xx2,lm1,lm2)=[phi1(xx1,lm1);phi2(xx2,lm2)]

FF(xx1,xx2,lm1,lm2,mu)=[f;phi(xx1,xx2,lm1,lm2);phi_s(xx1,xx2,mu)]


% %% Reformulated system


n=length(z)
x=sym('x',[1,n]);

psi=.5*norm(FF)^2
D0=eye(n)


FF=subs(FF,z,x)
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
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
fprintf('Just finished main-iteration #%d iteration #%d\n ',tttt,counter)
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
clearvars s
end
total_ttt
tt11=vpa(total_ttt,4)



