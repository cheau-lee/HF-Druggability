module load anaconda3
conda create -n drugnome_env python=3.6
conda config --append channels conda-forge 
conda activate drugnome_env

# Clone the DrugnomeAI GitHub repository into the local machine
git clone https://github.com/astrazeneca-cgr-publications/DrugnomeAI-release.git

# Override conflicts during installation
pip install --ignore-installed setuptools==39.1.0 numpy==1.16.3 numpydoc==0.8.0 pandas==0.24.2 scipy==1.2.1 scikit-learn==0.20.3 bokeh==1.1.0 h5py==2.9.0 tensorflow==1.12.0 Keras==2.2.4 matplotlib==3.0.3 palettable==3.1.1 plotly==3.9.0 PyYAML==5.1 seaborn==0.9.0 tables==3.5.1 twine==3.0.0 tqdm==4.14 umap-learn==0.3.8 xgboost==0.80 numba==0.37 llvmlite==0.22 grpcio==1.8.6 docutils==0.14 pillow==4.0

# Run the 'global_init.sh' script to perform any additional setup or initialisation required by DrugnomeAI 
sh global_init.sh

# Install the DrugnomeAI package using the setup script
python setup.py install
