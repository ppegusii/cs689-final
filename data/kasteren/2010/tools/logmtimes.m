function C = logmtimes(A,B)
% 
% [Da,Qa] = size(A);
% [Qb,Nb] = size(B);
% 
% A=A';
% C = zeros(Da,Nb);
% 
% for i=1:Da,
%     C(i,:) = logsum(repmat(A(:,i),1,Qa) + B,1);
% end


% 
[Da,Qa] = size(A);
[Qb,Nb] = size(B);

D = repmat(reshape(A',[Qa 1 Da]),[1 ,Nb 1]);
E = repmat(B, [1 1 Da]);

C = reshape(logsum(D+E,1),Nb,Da)';
