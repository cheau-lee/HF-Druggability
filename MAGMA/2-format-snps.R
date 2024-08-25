# MAGMA requires columns in a certain order with the correct names

# Read the tsv file

snps <- data.table::fread('/data/home/bt23020/HF_Project/HF_Project/HF-multiancestry-maf0.01.tsv')

# Select and rename columns

snps <- dplyr::select(snps, variant_id, p_value, chromosome, base_pair_location, other_allele, effect_allele)

# Create column of HF cases (taken from Levin paper)

snps$N <- 115150 
data.table::setnames(snps, c('p_value', 'variant_id'), c('P', 'SNP'))

# Write the processed data to a new file

data.table::fwrite(snps, '/data/home/bt23020/HF_Project/MAGMA_SNPs.txt', sep = '\t')
