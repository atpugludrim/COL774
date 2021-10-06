echo "Extracting class 2 & 3 from train.csv in train2.csv"
awk -F ',' '{if ( $NF==2 || $NF==3 ){print $0}}' $1 > train2.csv
echo "Extracting class 2 & 3 from test.csv in test2.csv"
awk -F ',' '{if ( $NF==2 || $NF==3 ){print $0}}' $2 > test2.csv
