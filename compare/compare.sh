#!/bin/bash


if [[ $# -ne 2 ]];then
   echo "argc=$#"
   echo "Use $0 file1.rsf file2.rsf"
   exit 1
fi



if file $1 | grep 'ASCII text';then
if file $2 | grep 'ASCII text';then
  cat $1 | sed -e "s/^in\=.*$/in=\".\/diff.rsf@\"/" > ./diff.rsf
  source $1
  [[ $n1 -gt 0 && $n2 -gt 0 && $n3 -gt 0 && $n4 -gt 0 ]] && ./compare.exe $n1 $n2 $n3 $n4 ${1}@ ${2}@ ./diff.rsf@ && exit 0
fi
fi

exit 1
