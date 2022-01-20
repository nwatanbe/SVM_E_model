SVM E-model Readme
by Naoki Watanabe, Christopher J. Vavricka and Michihiro Araki

1.  System requirements
- Any Linux operating system with Anaconda is recommended for use.
- The software was tested on CentOS 7.5.1804 (Core) with Anaconda3-2019.10.
Other packages were used as follows.
python 3.7.4
scikit-learn 0.21.3
numpy
scipy 
pandas

No non-standard hardware is required.

2. Installation guide
Anaconda are downloaded from https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh. Anaconda are installed:
./Downloaded_directory_path/Anaconda3-2019.10-Linux-x86_64.sh
Downloaded_directory_path should be given an arbitrary directory path containing Anaconda.

SVM_E-model.tar.gz.is downloaded from XXX. The files are extracted:
tar –xvf /Downloaded_directory_path/SVM_E-model.tar.gz

Conda environment are built:
conda create –n enzyme python=3.7.4
conda activate enzyme

scikit-learn installation:
conda install scikit-learn==0.21.3

pandas installation:
conda install pandas

The software typically takes several minutes to install.

3. Demo 
The directory including in SVM_E-model script is changed into:
cd /Downloaded_directory_path/SVM_E-model/script/
Downloaded_directory_path should be given an arbitrary directory path containing SVM_E-model.

SVM_E-model script is run. More accurate five CYP76AD classification model using small datasets are built based on cross validation results and some sample sequences are tested using the most accurate model:
python run_ml.py 

The outputs are constructed as follows (/Downloaded_directory_path/SVM_E-model/result/):
	More accurate five models
	Cross validation results in the five models
	Test results in the most accurate model

Run time in the demo is several minutes.

4. Instructions for use
An enzyme family classification (binary classification) models are enabled to build using your data. Positive and negative datasets are named sample_positive.vec and sample_negative.vec, respectively. Both files are in /Downloaded_directory_path/SVM_E-model/train/. Test dataset are named sample_test.vec. The test file is in /Downloaded_directory_path/SVM_E-model/test/.The enzyme sequence in training and test data should be converted into the vectors in the same feature extraction.  The run time depends on dataset size and the number of dimensions in vectors.

5 License
This software is released under the MIT License, see LICENSE.txt.
