function  getTheDataset()



match_data= dlmread("Match.csv");
ball_data= dlmread("Ball_by_Ball.csv");

match_data(1:10,:);
ball_data(1:10,:);


%Setting the data ready!!

X= zeros(size(ball_data,1), 10);
y=zeros(size(ball_data,1),1);

size(X);

prev=335987;
rno=1;

for i=1:size(X,1)

	match_id= ball_data(i,1);

	if(prev!=match_id)
		rno+=1;
	endif;

	prev=match_id;

	X(i,1)= 1; %Team Name
	X(i,2)= 2; %Opponent
	X(i,3)= match_data(rno,4); %Season_Id
	X(i,4)= (match_data(rno,3)==match_data(rno,5))+1; %Toss Winner
	X(i,5)= match_data(rno,6); %Toss decision
	X(i,6)= ball_data(i,2); %Innings
	X(i,7)= ball_data(i,3);  %Over
	X(i,8)= ball_data(i,4);	%Ball_by_Ball
	X(i,9)= ball_data(i,5);  %Runs
	X(i,10)= (ball_data(i,6)!=0);	%Wicket

	y(i,1)= (match_data(rno,3)==match_data(rno,7))+1;
end;

csvwrite("dataset.csv",X);
csvwrite("winner.csv",y);

end;
