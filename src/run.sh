#!/bin/sh
#NUMBER OF ARGUMENTS PASSED MUST BE EITHER 4 OR 5
nargs=$#
usage(){
	echo "Usage: $0 1 <train-path> <test-path> <part-num>" 1>&2;
	echo "Usage: $0 2 <train-path> <test-path> <binary or multi> <part-num>" 1>&2; exit 1;
}
if (( $nargs==4 ))
then
	if ! (( $1==1 ))
	then
		usage
	fi
	trainp=$2
	testp=$3
	partnum=$4
	[[ "$partnum" =~ ([a-e]|g) ]] && python3 -c "import nltk; nltk.download('stopwords')" || usage
	cd q1
	if [[ "$partnum" == "a" ]]
	then
		if ! [ -e train_x.txt -a -e train_y.txt -a -e test_x.txt -a -e test_y.txt ]
		then
			echo "Preprocessing data"
			python3 process_data.py -dpath "$trainp" -outpath train -quiet
			python3 process_data.py -dpath "$testp" -outpath test -quiet
		else
			echo "Using already existing file"
		fi
		echo "Running part Q1(a). This should take around 10 mins."
		echo "Training"
		python3 q1a.py -px "train_x.txt" -py "train_y.txt" --train --output_file thetas1a
		echo "Training complete"
		echo "Accuracy on train set"
		python3 q1a.py -px "train_x.txt" -py "train_y.txt" --thetas_file thetas1a
		echo "Accuracy on test set"
		python3 q1a.py -px "test_x.txt" -py "test_y.txt" --thetas_file thetas1a
	elif [[ "$partnum" == "b" ]]
	then
		if ! [ -e train_y.txt -a -e test_y.txt ]
		then
			echo "Preprocessing data"
			python3 process_data.py -dpath "$trainp" -outpath train -quiet
			python3 process_data.py -dpath "$testp" -outpath test -quiet
		else
			echo "Using already existing file"
		fi
		python3 -py test_y.txt -py_train train_y.txt
	elif [[ "$partnum" == "c" ]]
	then
		echo "This can only be run after Q1(a) is run."
		python3 q1c.py --output Conf1a
	elif [[ "$partnum" == "d" ]]
	then
		if ! [ -e train_stem_stop_x.txt -a -e train_stem_stop_y.txt -a -e test_stem_stop_x.txt -a -e test_stem_stop_y.txt ]
		then
			echo "Preprocessing data"
			python3 process_data.py -dpath "$trainp" -outpath train -quiet -stem-stop
			python3 process_data.py -dpath "$testp" -outpath test -quiet -stem-stop
		else
			echo "Using already existing file"
		fi
		echo "Running part Q1(d). This should take around 5-10 mins."
		echo "Training"
		python3 q1d.py -px "train_step_stop_x.txt" -py "train_step_stop_y.txt" --train --output_file thetas1d
		echo "Accuracy on test set"
		python3 q1d.py -px "test_step_stop_x.txt" -py "test_step_stop_y.txt" --thetas_file thetas1d
	elif [[ "$partnum" == "e" ]]
	then
		if ! [ -e train_stem_stop_x.txt -a -e train_stem_stop_y.txt -a -e test_stem_stop_x.txt -a -e test_stem_stop_y.txt ]
		then
			echo "Preprocessing data"
			python3 process_data.py -dpath "$trainp" -outpath train -quiet -stem-stop
			python3 process_data.py -dpath "$testp" -outpath test -quiet -stem-stop
		else
			echo "Using already existing file"
		fi
		echo "Running part Q1(e)."
		echo "Training"
		python3 q1e.py -px "train_step_stop_x.txt" -py "train_step_stop_y.txt" --train --output_file thetas1e
		echo "Accuracy on test set"
		python3 q1e.py -px "test_step_stop_x.txt" -py "test_step_stop_y.txt" --thetas_file thetas1e
	elif [[ "$partnum" == "g" ]]
	then
		echo "Running part Q1(g)."
		if ! [ -e summary_stem_stop_x.txt -a -e summary_test_stem_stop_x.txt ]
		then
			python3 process_data_for_g.py -dpath "$trianp" -stem-stop -outpath summary
			python3 process_data_for_g.py -dpath "$testp" -stem-stop -outpath summary_test
		fi
		python3 q1g.py -px "train_stem_stop_x.txt" -py "train_stem_stop_y.txt" --train --output_file thetas1g --thetas_file thetas1d -sx summary_stem_stop_x.txt
		python3 q1g.py -px "test_stem_stop_x.txt" -py "test_stem_stop_y.txt" --thetas_file thetas1g -sx summary_stem_stop_x.txt
	else
		usage
	fi
elif (( $nargs==5 ))
then
	if ! (( $1==2 ))
	then
		usage
	fi
	trainp=$2
	testp=$3
	bin_mult=$4
	partnum=$5
	cd q2
	if (( $bin_mult==0 ))
	then
		if [[ "$partnum" == "a" ]]
		then
			if ! [ -e train2.csv -a -e test2.csv ]
			then
				echo "This uses awk to extract binary classification samples."
				bash extract.sh "$trainp" "$testp"
			fi
			python3 q2a_vectorized.py --path-train train2.csv --savep --kernel "linear" --quiet
			python3 q2a_vectorized.py --path-test test2.csv --kernel "linear" --quiet
		elif [[ "$partnum" == "b" ]]
		then
			if ! [ -e train2.csv -a -e test2.csv ]
			then
				echo "This uses awk to extract binary classification samples."
				bash extract.sh "$trainp" "$testp"
			fi
			python3 q2a_vectorized.py --path-train train2.csv --savep --kernel "gaussian" --quiet
			python3 q2a_vectorized.py --path-test test2.csv --kernel "guassian" --quiet
		elif [[ "$partnum" == "c" ]]
		then
			if ! [ -e train2.csv -a -e test2.csv ]
			then
				echo "This uses awk to extract binary classification samples."
				bash extract.sh "$trainp" "$testp"
			fi
			echo "Using libsvm with linear kernel"
			python3 q2aiii.py --path-train train2.csv --kernel "linear" --quiet
			python3 q2aiii.py --path-test test2.csv --kernel "linear" --quiet
			echo "Using libsvm with gaussian kernel"
			python3 q2aiii.py --path-train train2.csv --kernel "gaussian" --quiet
			python3 q2aiii.py --path-test test2.csv --kernel "gaussian" --quiet
		else
			usage
		fi
	elif (( $bin_mult==1 ))
	then
		if [[ "$partnum" == "a" ]]
		then
			python3 q2b_vectorized.py --path-train "$trainp" --savep --quiet
			python3 q2b_vectorized.py --path-test "$testp" --quiet
		elif [[ "$partnum" == "b" ]]
		then
			python3 q2bii.py --path-train "$trainp" --quiet
			python3 q2bii.py --path-test "$testp" --quiet
		elif [[ "$partnum" == "c" ]]
		then
			echo "This can only be run after the first two parts for multiclass are run."
			echo "For cvxopt"
			python3 q2biii.py --output "conf_cvx"
			echo "For libsvm"
			python3 q2biii.py --output "conf_libsvm" --libsvm
		elif [[ "$partnum" == "d" ]]
		then
			echo "This will take around one hour."
			python3 q2biv.py -train "$trainp" -test "$testp"
		else
			usage
		fi
	else
		usage
	fi
else
	usage
fi
