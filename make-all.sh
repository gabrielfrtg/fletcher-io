#!/bin/bash

cd original
make clean; make -j8
cd ../original-parallel-write-single-file
make clean; make -j8
cd ../original-parallel-write-part-files
make clean; make -j8