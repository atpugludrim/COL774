echo "Extracting class 2 & 3 from train.csv in train2.csv"
awk -F ',' '{if ( $NF==2 || $NF==3 ){print $0}}' train.csv > train2.csv
echo "Extracting class 2 & 3 from test.csv in test2.csv"
awk -F ',' '{if ( $NF==2 || $NF==3 ){print $0}}' test.csv > test2.csv
