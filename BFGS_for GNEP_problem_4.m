clear
clc

%%%%%%%%%%%  Parameters values

N=4            %Number of players
M=1             % Number of shared constraints here only one 
mm=100
rrrr=2
err=10^-8
c=.01
c1=c/3
theta=1-c
n=M+2*N         %Number of variables

BBBB=1          % Buffer capacity
lv=.01          % Minimal amount of data to be sent

beta=10^-8
alpha1=.01
alpha2=3    
syms armizo WWP



x=sym('x',[1,n]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%            Problem
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%           This programme to calculate GNEP with "N" players 


%%%%          with Buffer capacity "BBBB" and lower limit of data to be sent "l_v"



             


syms(sym('xx',[1 N]))
xx=sym('xx',[1 N])
syms(sym('lm',[1 N]))
lm=sym('lm',[1 N])
syms(sym('mu',[1,M]))
mu=sym('mu',[1,M])
theta00=sym('theta0',[1,N])
zzz=[xx lm mu]


%%% objective functions

for i=1:N
    theta0(i)=(xx(i)/BBBB) -(xx(i)/sum(xx)) 
end
    


% player constraints

for i=1:N
h(i)=(lv-xx(i))          
end

% shared constraint

s=sum(xx)-BBBB

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%  Lagrangian and then it's Gradient too

 


for i=1:N  % Gradient of Lagrangian for every player
    l(i)=gradient(theta0(i)+lm(i)*h(i)+mu*s,xx(i))    
end


ps=@(xx1,xx2) sqrt(xx1^2+xx2^2)-(xx1+xx2)
pss(xx1,xx2)=ps(xx1,xx2)^2


kkk(xx1,xx2)=pss(xx1,xx2)

syms(sym('phi',[1,N]))
syms(sym('phi_s',[1,M]))
phi_s0=sym('phi_s',[1,M])
phi0=sym('phi',[1,N])
 


for i=1:N
    phi(i)=kkk(-h(i),lm(i))   
end

for i=1:M
    phi_s(i,:)=kkk(-s(i),mu(i))
end

f=transpose([l(1:N)])


phi=phi.'

FF=[f;phi;phi_s]  % Reformulated system



psi=.5*norm(FF)^2
D0=eye(n)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Initial value



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Selection of Method Armizo or Goldstein method

        %method0=armizo
        method0=WWP


 for tttt=1:25
    if tttt<6
   a=.36+rand(1,9)*.01
   norm(a)
    end

    if tttt<11 & tttt>=6
   a=.43+rand(1,9)*.01
   norm(a)
    end

    if tttt<16 & tttt>=11
   a=.16+rand(1,9)*.01
   norm(a)
    end

    if tttt<21 & tttt>=16
  a=.22+rand(1,9)*.01
   norm(a)
    end

    if tttt>=21
   a=.36+rand(1,9)*.01
   norm(a)
    end


 
   
tic

b=a;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FF=subs(FF,[xx lm mu],x)
F=subs(psi,[xx lm mu],x)
f0=subs(FF,x,a)

F0=vpa(subs(F,x,a));
G=gradient(F);
G0=vpa(subs(G,x,a));

D0=eye(n);
counter=1;

%%%%%%  Main algorithm starts

  

while  norm(G0)>err  %%%%  Stopping condition
     

d=-D0^-1*G0;                  %%% Descent direction
%d=-D0\G0
if d=='Fail'
    break;
end

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
  
if s>200
        disp('Value of step size is higher than 200')
        fprintf('We were in iteration #%d\n',counter)
        break;
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

if norm(G0)>=1
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
 if ee(1)<0 | ee(2) <0 | ee(3) <0 |ee(4)<0 |ee(5) <0 | ee(6) <0 |ee(7)<0
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
fprintf('Just finished main iteration#%d and  #%d\n ',tttt,counter)
counter=counter+1;
disp('Norm of gradient')
vpa(norm(G0),2)
disp('Norm of function')
vpa(norm(f0),2)
%fprintf('<strong> BFGS by %s \n</strong>',method0)
f0=subs(FF,x,a);
if counter>500
    break;
end

end

syms iter  step_chosen step_size norm_of_fun norm_of_grad;
fprintf('Total number of Iterations : #%d\n',counter-1)
[iter step_chosen step_size norm_of_fun norm_of_grad;m]
disp('initial value was: ')
b
distance=norm(b)

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
t=toc;

total_ttt(tttt,:)=vpa([tttt distance t counter-1 vpa(F0,2) G2 a 00000 b],4)
tttt=tttt+1;
end
total_ttt
tt11=vpa(total_ttt,6)