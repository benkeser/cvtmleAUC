#-----------------------------------------
# commands for scp'ing sce and cent over
#-----------------------------------------
cd ~/Dropbox/R/cvtmleAUC/sandbox
scp cent_new.R sce_new.sh makeData.R dbenkese@snail.fhcrc.org:~/cvtmleauc
scp cent_newest.R sce.sh makeData.R dbenkese@snail.fhcrc.org:~/cvtmleauc

ssh dbenkese@snail.fhcrc.org
cd cvtmleauc
scp cent_new.R sce_new.sh makeData.R dbenkese@rhino.fhcrc.org:~/cvtmleauc
scp cent_newest.R sce.sh makeData.R dbenkese@rhino.fhcrc.org:~/cvtmleauc

ssh dbenkese@rhino.fhcrc.org
cd cvtmleauc
chmod +x cent* sce*
ml R
./sce_new.sh ./cent_new.R small_v2
./sce.sh ./cent_newest.R runtn_full_v1

cd ~/Dropbox/R/cvtmleAUC/sandbox
scp ../R/wrapper_functions.R cent_oracles.R sce_oracles.sh makeData.R dbenkese@snail.fhcrc.org:~/cvtmleauc
ssh dbenkese@snail.fhcrc.org
cd cvtmleauc
scp wrapper_functions.R cent_oracles.R sce_oracles.sh makeData.R dbenkese@rhino.fhcrc.org:~/cvtmleauc
ssh dbenkese@rhino.fhcrc.org
cd cvtmleauc
chmod +x cent_oracles* sce_oracles*
ml R
./sce_oracles.sh ./cent_oracles.R oracles_v15


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
cd ~/Dropbox/R/cvtmleauc/sandbox/simulation
scp dbenkese@snail.fhcrc.org:~/cvtmleauc/allOut.RData . 

#-----------------------------------------
# misc commands 
#-----------------------------------------
squeue -u dbenkese
# scancel `seq 51239645 51239655`
