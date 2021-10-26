#!/bin/bash
#NUMBER OF ARGUMENTS PASSED MUST BE EITHER 4 OR 5
nargs=$#
usage(){
	echo "Usage: $0 1 <train-path> <val-path> <test-path> <part-num>" 1>&2;
	echo "Usage: $0 2 <train-path> <test-path> <part-num>" 1>&2; exit 1;
}
if (( $nargs==5 ))
then
	if ! (( $1==1 ))
	then
		usage
	fi
	trainp=$2
	valp=$3
	testp=$4
	partnum=$5
	[[ "$partnum" =~ ([a-d]) ]] || usage
	if [[ "$partnum" == "a" ]]
	then
		echo "Training"
		python3 q1a_mod_acc_data.py "$trainp" "$valp" "$testp"
		echo "A png file of graph should be saved in current directory"
		python3 q1a_mod_onehot_acc_data.py "$trainp" "$valp" "$testp"
		echo "Another png file of graph should be saved in current directory"
	elif [[ "$partnum" == "b" ]]
	then
		python3 q1b.py "$trainp" "$valp" "$testp"
		echo "Best model and params should have been saved in a pickle file"
	elif [[ "$partnum" == "c" ]]
	then
		python3 q1c.py "$trainp" "$valp" "$testp"
	elif [[ "$partnum" == "d" ]]
	then
		if ! [ -e best_params.pkl ]
		then
			python3 q1c.py "$trainp" "$valp" "$testp"
		fi
		python3 q1d.py "$trainp" "$valp" "$testp"
	else
		usage
	fi
elif (( $nargs==4 ))
then
	if ! (( $1==2 ))
	then
		usage
	fi
	trainp=$2
	testp=$3
	partnum=$4
	[[ "$partnum" =~ ([a-g]) ]] || usage
	[[ "$partnum" =~ ([c-f]) ]] && ! [ -e test.csv -a -e train.csv ] && python3 q2a.py "$trainp" "$testp"
	if [[ "$partnum" == "a" ]]
	then
		python3 q2a.py "$trainp" "$testp"
	elif [[ "$partnum" == "b" ]]
	then
		echo "Code written, see util.py and q2b.py"
	elif [[ "$partnum" == "c" ]]
	then
		python3 q2c.py
	elif [[ "$partnum" == "d" ]]
	then
		python3 q2d.py
	elif [[ "$partnum" == "e" ]]
	then
		python3 q2e.py
	elif [[ "$partnum" == "f" ]]
	then
		python3 q2f.py
	elif [[ "$partnum" == "g" ]]
	then
		python3 q2g.py "$trainp" "$testp"
		python3 q2g_classifier.py
	else
		usage
	fi
else
	usage
fi
