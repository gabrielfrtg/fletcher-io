#!/bin/bash


if [[ $# -ne 1 ]];then
   echo "argc=$#"
   echo "Use $0 file1.rsf"
   exit 1
fi



if file $1 | grep 'ASCII text';then
  cat $1 | sed -e "s/^in\=.*$/in=\".\/diff.rsf@\"/" > ./diff.rsf
  source $1
  f2=$((n2 / 2))
  [[ $n1 -gt 0 && $n2 -gt 0 && $n3 -gt 0 && $n4 -gt 0 ]] && sfwindow \
	  < $1 f2=$f2 n2=1 | sfgrey gainpanel=all wantframenum=no polarity=y | sfpen

fi
