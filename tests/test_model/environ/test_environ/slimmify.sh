#!/bin/bash
ls 1environ.* | awk '{if (NR%2==0) if (substr($1,length($1))=="*") print substr($1,1,length($1)-1); else print $1}' > rmfiles.txt
xargs rm < rmfiles.txt
rm -f rmfiles.txt
