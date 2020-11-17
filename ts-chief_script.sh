#!/bin/bash
n_cv=9
n_r=4
size=1000

for id_r in `seq 1 $n_r`
do
	for id_cv in `seq 0 $n_cv`
	do
		jdk/jdk-15/bin/java -jar tschief.jar -train="datasets/TSCHIEF/data_Train_"$size"_"$id_cv"_R"$id_r".csv" -test="datasets/TSCHIEF/data_Test_"$size"_"$id_cv"_R"$id_r".csv" -out="results/TSCHIEF/" -repeats="1" -trees="500" -s="ee:10,boss:150,rise:150" -export="1" -verbosity="1" -shuffle="True" -target_column="last"
	done
done

