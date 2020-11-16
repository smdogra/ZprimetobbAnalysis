#!/bin/bash
CARDDIR=$PWD
echo $CARDDIR
i=1

while IFS= read -r line
do
    test $i -eq 1 && ((i=i+1)) && continue
    
    zmass=$(echo $line | awk -F " " '{print$1}')
    dbs=$(echo $line | awk -F " " '{print$2}')
    echo  $zmass $dbs
    zmass_=$(echo ${zmass} | sed -e 's/\./p/' -e 's/-/m/'  -e 's/-/m/')
    dbs_=$(echo ${dbs} | sed -e 's/\./p/' -e 's/-/m/'  -e 's/-/m/')
    
    ############# Process Name 

    CARDLABEL1=ZprimeTobb${zmass_}_dbs${dbs_}
    CARDNAME1=${CARDLABEL1}
    DIR1=$CARDDIR/${CARDLABEL1}
    mkdir -p $DIR1
    cp $CARDDIR/extramodels.dat $DIR1/${CARDNAME1}_extramodels.dat
    cp $CARDDIR/run_card.dat $DIR1/${CARDNAME1}_run_card.dat
    sed -e "s/_massZ_/$zmass/" -e "s/_dbs_/$dbs/"   $CARDDIR/customizecard
s.dat > $DIR1/${CARDNAME1}_customizecards.dat
    sed "s/_NAME_/$CARDNAME1/" $CARDDIR/proc_card.dat > $DIR1/${CARDNAME1}
_proc_card.dat
        
done < massgrid.txt
