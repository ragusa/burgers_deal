clear all; close all; clc;

M=load('matrix.txt');

ni=max(M(:,1))+1;
nj=max(M(:,2))+1;
if(ni~=nj)
    error('wrong matrix size ...')
end

nnz = length(M(:,1));

A=spalloc(ni,nj,nnz);

for k=1:nnz
    i=M(k,1)+1;
    j=M(k,2)+1;
    A(i,j)=M(k,3);
end

condest(A)

figure(1)
spy(A);

b=ones(ni,1);
x=A\b;
figure(2);
plot(x)
