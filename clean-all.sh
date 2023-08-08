#!/bin/bash

rm ./TTI*

cd original
make clean ; rm TTI*
cd ../original-parallel-write-single-file
make clean ; rm TTI*
cd ../original-parallel-write-part-files
make clean ; rm TTI*

