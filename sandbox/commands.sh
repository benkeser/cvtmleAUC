#-----------------------------------------
# commands for scp'ing sce and cent over
#-----------------------------------------
cd ~/Dropbox/R/cvtmleAUC/sandbox
scp cent.R sce.sh makeData.R dbenkese@snail.fhcrc.org:~/cvtmleauc

ssh dbenkese@snail.fhcrc.org
cd cvtmleauc
scp cent.R sce.sh makeData.R dbenkese@rhino.fhcrc.org:~/cvtmleauc

ssh dbenkese@rhino.fhcrc.org
cd cvtmleauc
chmod +x cent* sce*
./sce.sh ./cent.R run_v5

#-----------------------------------------
# commands to get into rhino and load R
#-----------------------------------------
ssh dbenkese@snail.fhcrc.org
ssh dbenkese@rhino.fhcrc.org
 # enter password
ml R/3.2.0
R
# module avail
#-----------------------------------------
# scp results from rhino to local machine
#-----------------------------------------
# from rhino
cd cvtmleauc/out
scp allOut.RData dbenkese@snail.fhcrc.org:~/cvtmleauc
	# enter snail password
 	# ctrl + shift + t to open up new term
# scp to snail
cd ~/Dropbox/Emory/cvtmleaucSieve/simulation
scp dbenkese@snail.fhcrc.org:~/cvtmleauc/allOut.RData . 

#-----------------------------------------
# misc commands 
#-----------------------------------------
squeue -u dbenkese
# scancel `seq 51239645 51239655`
