clear all                   % this is always good to do, because it helps finding errors in your code.
rand('state',0);            % to test your program, it is often good to fix the random seed so you can compare outcome of runs while debugging
randn('state',0);
n=20;                        % number of spins
Jth=0.1;                    % Jth sets the size of the random threshold values th
if 0, % toggle between full and sparse Ising network
    % full weight matrix
    J0=0;                        % J0 and J are as defined for the SK model
    J=0.5;
    w=J0/n+J/sqrt(n)*randn(n,n);
    w=w-diag(diag(w));
    w=tril(w)+tril(w)';
    c=~(w==0);                  % neighborhood graph fully connected
else
    % sparse weight matrix
    c1=0.5;                          % connectivity is the approximate fraction of non-zero links in the random graph on n spins
    k=c1*n;
    beta=0.5;
    w=sprandsym(n,c1);          % symmetric weight matrix w with c1*n^2 non-zero elements
    w=w-diag(diag(w));
    c=~(w==0);                  % sparse 0,1 neighborhood graph 
    w=beta*((w>0)-(w<0));              % w is sparse with +/-beta on the links
end;
th = randn(n,1)*Jth ;

%EXACT
sa= s_all(n) ;              % all 2^n spin configurations
Ea = 0.5 *sum(sa.*(w*sa')',2) + sa*th; % array of the energies of all 2^n configurations
Ea=exp(Ea); 
Z=sum(Ea); 
p_ex=Ea /Z ;                % probabilities of all 2^n configurations
m_ex=sa' *p_ex;             % exact mean values of n spins
klad=(p_ex*ones(1,n)).*sa;
chi_ex=sa'*klad-m_ex*m_ex'; % exact connected correlations

%MF
%write your code
m_mf=m;
error_mf=sqrt(1/n*sum(m_mf-m_ex).^2)

%BP
%write your code
error_bp=sqrt(1/n*sum(m_bp-m_ex).^2)

