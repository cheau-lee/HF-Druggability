#!/bin/bash
#$ -cwd           
#$ -pe smp 5      
#$ -l h_rt=240:0:0  
#$ -l h_vmem=10G   
#$ -m bea

module load anaconda3
conda activate drugnome_env
cd /data/home/bt23020/HF_Project/DrugnomeAI-release


# Normal setting with all models
drugnomeai \
  -c /data/home/bt23020/HF_Project/DrugnomeAI-release/drugnome_ai/conf/hf_config.yaml \
  -o FDR_normal \
  -m \
  -l \
  -k /data/home/bt23020/HF_Project/FDR_Genes.txt \

  
# Top performing XGBoost model
  drugnomeai -c /data/home/bt23020/HF_Project/DrugnomeAI-release/drugnome_ai/conf/hf_config.yaml -o FDR_xg_norm_results --superv-models xgb -m -l -k /data/home/bt23020/HF_Project/FDR_Genes.txt
