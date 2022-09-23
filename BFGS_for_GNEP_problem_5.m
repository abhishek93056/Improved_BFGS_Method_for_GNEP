clear
clc

%%%% This programme to calculate GNEP with "N" players 
%%%% with Buffer capacity "BBBB" and lower limit of data to be sent "l_v"

for tttt=1:50

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Selection of Method Armizo or Goldstein method

syms armizo WWP
%method0=armizo
method0=WWP


%%%%%     Starting points
if tttt<10
   a=.05+rand(1,8)*.01
   norm(a)
end
if tttt<20 & tttt >= 10
   a=.1+rand(1,8)*.01
   norm(a)
end
if tttt<30 & tttt>=20
   a=.17+rand(1,8)*.01
   norm(a)
end
if tttt<40 & tttt>=30
   a=.25+rand(1,8)*.01
   norm(a)
end
if tttt>=40
   a=.31+rand(1,8)*.01
   norm(a)
end
   
    
N=3    %Number of players
M=1     % Number of shared constraints here only one 


mm=100
rrrr=2
err=10^-8
c=.01
c1=c/3
theta=1-c
n=M+2*N+1           %Number of variables

BBBB=1              % Buffer capacity
lv=.01              % Minimal amount of data to be sent

beta=10^-8
alpha1=.01
alpha2=3


bbbb=rand(1,n)



%%%%%%%  Time start
tic

b=a;
x=sym('x',[1,n]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%            Problem
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



             


syms(sym('xx',[1 N]))
xx=sym('xx',[1 N])
syms(sym('lm',[1 N+1]))  %  2 extra variable in player 1

%%%% lm1 and lm2 for player 1

lm=sym('lm',[1 N+1])
syms(sym('mu',[1,M]))
mu=sym('mu',[1,M])
theta00=sym('theta0',[1,N])
zzz=[xx lm mu]

for i=1:N
    theta0(i)=(xx(i)/BBBB) -(xx(i)/sum(xx)) %%% objective functions
end
   
h11=.3-xx(1)
h12=xx(1)-.5
hh=[h11 h12]

for i=2:N
h0(i)=(lv-xx(i))  % player constraints
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h0(:,1)=[]
h=[hh h0]


s=sum(xx)-BBBB

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%  Lagrangian and then it's Gradient too

 
l(1)=gradient(theta0(1)+lm(:,1:2)*h(:,1:2).'+mu*s,xx(1))

for i=2:N
    l(i)=gradient(theta0(i)+lm(i+1)*h(i)+mu*s,xx(i)) 
    
end

%%%%%%%%%%%%%%%%%%       Complementarity function
ps=@(xx1,xx2) sqrt(xx1^2+xx2^2)-(xx1+xx2)
pss(xx1,xx2)=ps(xx1,xx2)^2
kkk(xx1,xx2)=pss(xx1,xx2)



syms(sym('phi',[1,N+1]))        % For the  player 1
phi0=sym('phi',[1,N+1])
syms(sym('phi_s',[1,M]))
phi_s0=sym('phi_s',[1,M])


%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% %%%%%%%%      Objective function computation

for i=1:N+1
    phi(i)=kkk(-h(i),lm(i))   
end

for i=1:M
    phi_s(i,:)=kkk(-s(i),mu(i))
end





f=transpose([l(1:N)])
phi=phi.'

FF=[f;phi;phi_s]            % Reformulated system


psi=.5*norm(FF)^2
FF(xx,lm,mu)=@(xx,lm,mu)FF


D0=eye(n)

FF=subs(FF,[xx lm mu],x)
F=subs(psi,[xx lm mu],x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F0=(subs(F,x,a))
G=gradient(F)
G0=vpa(subs(G,x,a))

D0=eye(n);
counter=1;
  
%%%%%%%%        Main Algorithm starts
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %%%%%%%%      Stopping condition is     norm(G0)>err

while norm(G0)>err  
     

d=-D0^-1*G0;



%%%%%%%%%%%%%%%     Descent direction test
test_for_descent_dir=vpa((transpose(G0)*d)); 
if test_for_descent_dir>=0
    disp('d is not a descent direction')
    break;
end
    


%%%%%%%%%%%                 Computation of steplength


syms s;
z=a+mm*(rrrr)^-s*d' ;                %%%%%% z_k+ alpha d_k             

F_new(s)=vpa(subs(F,x,z));
G_new(s)=vpa(subs(G,x,z));

s=0;

if method0==armizo

while F_new(s)> F0 + c*mm*rrrr^-s*G0.'*d | F_new(s)< F0 + (1-c)*mm*rrrr^-s*G0.'*d 

    s=s+1;
    if s>200
        disp('Value of step size is higher than 150')
        fprintf('We were in iteration #%d\n',counter)
        break;
    end
end
end

if method0==WWP

while F_new(s)> F0 + c*mm*rrrr^-s*G0'*d +mm*rrrr^-s*min(-c1*G0'*d,(c*mm*rrrr^-s*norm(d)^2)/2) | G_new(s)'*d< theta*G0'*d + min(-c1*G0'*d,(c*mm*rrrr^-s*norm(d)^2))
s=s+1;
    if s>200
        disp('Value of step size is higher than 150')
        fprintf('We were in iteration #%d\n',counter)
        break;
    end
end
end
 s_1=s;
 disp(s_1);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('value of step size')
sss=mm*rrrr^-s;
disp(sss)


z=a+mm*rrrr^-s*d.';  %%%%%%%%%%%%%%%%%%% Updated value of "a"

if max(abs(a(1,1:N)-z(1,1:N)))<err                    
    disp('Root value is repeating')
    break;
end




%%%%%%%%%%%%%%%%%%%%%%%%%    BFGS update matrix


F0=vpa(subs(F,x,a));
F1=vpa(subs(F,x,z));
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

D0=vpa(D1);
quasi=abs(vpa(D1*s1)-vpa(yk));
if abs(vpa(D1*s1)-vpa(yk))<=err^2
    disp('Quasi equation satisfied')
else
    disp('Quasi Newton equation not satisfied')
    break;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%    Next Iteration update
a=z;
F0=(subs(F,x,a));

G0=(subs(G,x,a));
G2=(norm(G0));
m(counter,:)=[counter s_1 sss vpa(norm(F0),4) vpa(G2,4)];
fprintf('Just finished main-iteration #%d with iteration #%d\n ',tttt,counter)
counter=counter+1;
disp('Norm of gradient')
vpa(norm(G0),2)
disp('Norm of function')
vpa((F0),2)
if counter>1000
    disp('Iteration reached to its max')
    break;
end

end


syms iter  step_chosen step_size norm_of_fun norm_of_grad;
fprintf('Total number of Iterations : #%d\n',counter-1)
[iter step_chosen step_size norm_of_fun norm_of_grad;m]
disp('initial vaue was: ')
b
disp('Functional value')
subs(FF,x,a)

disp('Root of the function:')
vpa(a,4)
disp('Norm of  Gradient')
G0=(subs(G,x,a));
G2=vpa(norm(G0),2)
disp('Functional value')
vpa(subs(F,x,a),2)
fprintf('<strong> BFGS by %s   in iterations %d \n </strong>',method0,counter-1)
t=toc
distance=norm(b)
total_ttt(tttt,:)=vpa([tttt distance t counter-1 vpa(F0,2) G2 a 00000 b],4)
tttt=tttt+1;
clearvars -except tttt total_ttt 

end
total_ttt 
tt11=vpa(total_ttt,7)

