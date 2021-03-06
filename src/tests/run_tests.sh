#!/bin/bash

for filename in './'*.cpp; do
  fpath=${filename%/*}
  cppfile=${filename##*/}
  execfile=${cppfile%%.*}
  if [ $cppfile == "tests_main.cpp" ]; then
    continue
  fi
  echo -e "\nCompiling $filename to $execfile"
  g++ --std=c++11 tests_main.cpp $cppfile -o $execfile
  echo -e "\nRunning $execfile"
  $fpath/$execfile
done
