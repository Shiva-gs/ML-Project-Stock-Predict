#!/bin/bash

echo "date,open,high,low,close,volume,Adj Close,Name" > utilities_5yr.csv
cd utilities_5yr
files=$(ls *.csv)
for file in $files
do
	tail -n +2 $file >> ../utilities_5yr.csv
done