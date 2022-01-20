# SVM E-model Readme
by Naoki Watanabe, Christopher J. Vavricka and Michihiro Araki

1.  System requirements
2.  Installation guide
3.  Demo 
4.  Instructions for use
5.  License

# 1.  System requirements
Any Linux operating system with Anaconda is recommended for use.
The software was tested on CentOS 7.5.1804 (Core) with Anaconda3-2019.10.
Other packages were used as follows:
python 3.7.4
scikit-learn 0.21.3
numpy
scipy 
pandas

No non-standard hardware is required.

# 2.  Installation guide
Anaconda can be downloaded from the following link:
https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

Anaconda can be installed using the following command:
./Downloaded_directory_path/Anaconda3-2019.10-Linux-x86_64.sh

"Downloaded_directory_path" should be given an arbitrary directory path containing Anaconda.

SVM_E_model.is downloaded from the following link: https://github.com/nwatanbe/SVM_E_model

Conda environment are built using the following commands:
conda create –n enzyme python=3.7.4
conda activate enzyme

scikit-learn is installed as follows:
conda install scikit-learn==0.21.3

pandas is installed as follows:
conda install pandas

The software typically takes several minutes to install.

# 3.  Demo 
The directory including in SVM_E-model script is changed into:
cd /Downloaded_directory_path/SVM_E-model/script/
Downloaded_directory_path should be given an arbitrary directory path containing SVM_E-model.

SVM_E-model script is run for prediction of CYP76AD sequences with a simplified data set, by the following command:
python run_ml.py 

The following results are output into"/Downloaded_directory_path/SVM_E-model/result/":  
●	More accurate five models  
●	Cross validation results in the five models  
●	Test results in the most accurate model  

The demo run time is several minutes.

# 4.  Instructions for use
Enzyme family classification (binary classification) models are enabled to build using your data. Enzyme sequences for training and test data should first be converted into vectors, all using the same feature extractions. Positive and negative datasets should be named "sample_positive.vec" and "sample_negative.vec", respectively. Both files should be found in "/Downloaded_directory_path/SVM_E-model/train/". The test dataset file should be  named "sample_test.vec", and found in "/Downloaded_directory_path/SVM_E-model/test/".

Run time depends on the dataset size and the number of vector dimensions.

# 5.  License
This software is released under the MIT License, according to LICENSE.txt.

# 6. Reference
[1]Vavricka, C.J., et al., ‘Machine learning discovery of missing links that mediate alternative branches to plant alkaloids’ (in press)
