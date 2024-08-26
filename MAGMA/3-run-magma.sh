#!/bin/bash
#$ -cwd             # Set the working directory for the job to the current directory
#$ -pe smp 4        # Request 4 cores
#$ -l h_rt=1:0:0    # Request 1 hour runtime
#$ -l h_vmem=4G     # Request 4GB RAM per core

module load anaconda3
conda activate magma_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Annotate SNPs

/data/home/bt23020/HF_Project/MAGMA/magma \
    --annotate \
    --snp-loc /data/home/bt23020/HF_Project/HF-multiancestry-maf0.01.tsv \
    --gene-loc /data/home/bt23020/HF_Project/MAGMA/magma_data/NCBI37.3.gene.loc \
    --out /data/home/bt23020/HF_Project/MAGMA/magma_output/magma_annot_2.txt

# Gene analysis

/data/home/bt23020/HF_Project/MAGMA/magma \
    --bfile /data/home/bt23020/HF_Project/MAGMA/magma_data/g1000_eur \
    --gene-annot /data/home/bt23020/HF_Project/MAGMA/magma_output/magma_annot_2.txt.genes.annot \
    --pval /data/home/bt23020/HF_Project/MAGMASNPs_V2_converted.txt ncol=N \
    --gene-model snp-wise=mean \
    --out /data/home/bt23020/HF_Project/MAGMA/magma_output/magma_results_2.txt
