#!/bin/bash

echo "Run Handin Timo Marchand"

echo "Clearing/creating the plotting directory"
if [ ! -d "plots" ]; then
  mkdir plots
fi
rm -rf plots/*

echo "Download data points for Q2"
if [ ! -e vandermonde.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt
fi

# Question 1: Poisson Distribution
echo "Run the first script ..."
python3 NUR_Handin1_Q1.py > poisson.txt

# Question 2: Interpolation
echo "Run the second script ..."
python3 NUR_Handin1_Q2.py > interpolation.txt

echo "Generating the pdf"

pdflatex marchand.tex
bibtex marchand.aux
pdflatex marchand.tex
pdflatex marchand.tex


