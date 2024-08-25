# Create env for MAGMA

module load anaconda3
conda create -n magma_env
conda activate magma_env
conda install -c conda-forge libstdcxx-ng=9.3.0

# Installation

mkdir MAGMA
cd MAGMA
wget -O magma_v1.10.zip https://vu.data.surfsara.nl/index.php/s/zkKbNeNOZAhFXZB/download
unzip /data/home/bt23020/HF_Project/MAGMA/magma_v1.10.zip

# Test that MAGMA successfully installed by checking its version (should print MAGMA version: v1.10 (linux))

/data/home/bt23020/HF_Project/MAGMA/magma --version

# Obtain GRCh37 files from MAGMA site (https://cncr.nl/research/magma/) 

mkdir magma_data
file_path="/data/home/bt23020/HF_Project/MAGMA/magma_data/"
wget -O  ${file_path}NCBI37.3.zip https://vu.data.surfsara.nl/index.php/s/Pj2orwuF2JYyKxq/download
wget -O  ${file_path}g1000_eur.zip https://vu.data.surfsara.nl/index.php/s/VZNByNwpD8qqINe/download
wget -O  ${file_path}dbsnp151.synonyms.zip https://vu.data.surfsara.nl/index.php/s/MSeFJuAVKJ4HLHv/download

# Unzip the files in the same location

cd /data/home/bt23020/HF_Project/MAGMA/magma_data
unzip ${file_path}g1000_eur.zip -d ${file_path}
unzip ${file_path}NCBI37.3.zip -d ${file_path}
unzip ${file_path}dbsnp151.synonyms.zip -d ${file_path}
