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
		echo "Running part Q1(a). This should take around 30 mins."
		echo "Training"
		python3 q1a.py -p "$trainp" --train --output_file thetas1a
		echo "Training complete"
		echo "Accuracy on train set"
		python3 q1a.py -p "$trainp" --thetas_file thetas1a
		echo "Accuracy on test set"
		python3 q1a.py -p "$testp" --thetas_file thetas1a
	elif [[ "$partnum" == "b" ]]
	then
		echo b
	elif [[ "$partnum" == "c" ]]
	then
		echo "This can only be run after Q1(a) is run."
		python3 q1c.py --output Conf1a
	elif [[ "$partnum" == "d" ]]
	then
		echo "Running part Q1(d). This should take around 30 mins."
		echo "Training"
		python3 q1d.py -p "$trainp" --train --output_file thetas1d
		echo "Testing"
		python3 q1d.py -p "$testp" --thetas_file thetas1d
	elif [[ "$partnum" == "e" ]]
	then
		echo "Running part Q1(e)."
		echo "Training"
		python3 q1e.py -p "$trainp" --train --output_file thetas1e
		echo "Testing"
		python3 q1e.py -p "$testp" --thetas_file thetas1e
	elif [[ "$partnum" == "g" ]]
	then
		echo "Running part Q1(g)."
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
			# MAKE CLASS 2 vs 3 DATASET HERE
			python3 q2a_vectorized.py --path-train "$trainp" --savep --kernel "linear" --quiet
			python3 q2a_vectorized.py --path-test "$testp" --kernel "linear" --quiet
		elif [[ "$partnum" == "b" ]]
		then
			python3 q2a_vectorized.py --path-train "$trainp" --savep --kernel "gaussian" --quiet
			python3 q2a_vectorized.py --path-test "$testp" --kernel "guassian" --quiet
		elif [[ "$partnum" == "c" ]]
		then
			echo "Using libsvm with linear kernel"
			python3 q2aiii.py --path-train "$trainp" --kernel "linear" --quiet
			python3 q2aiii.py --path-test "$testp" --kernel "linear" --quiet
			echo "Using libsvm with gaussian kernel"
			python3 q2aiii.py --path-train "$trainp" --kernel "gaussian" --quiet
			python3 q2aiii.py --path-test "$testp" --kernel "guassian" --quiet
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
