set -beEuo pipefail

BASE_PATH=/users/amtseng/att_priors/data/raw/DREAM/
DATA_TO_USE_PATH=$BASE_PATH/DREAM_data_to_use.tsv
DESTINATION_BASE=$BASE_PATH
SYNAPSE_DOWNLOADER_PATH=/users/amtseng/tfmodisco/src/data/download_from_synapse.py

echo "Login to Synapse:"
printf "Username: "
read username
printf "Password: "
read -s password
printf "\n"

while read -r line
do
	outpath=$(printf "$line" | cut -f 3)
	synid=$(printf "$line" | cut -f 4)
	python $SYNAPSE_DOWNLOADER_PATH -u $username -p $password -d $DESTINATION_BASE/$outpath $synid
done <<< `cat $DATA_TO_USE_PATH | awk 'NR > 1'`
