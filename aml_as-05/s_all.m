function ss=s_all(n)
%function ss=s_all(n)
N=2^n;
ss=2*(double(dec2bin(0:N-1))-48)-1;
