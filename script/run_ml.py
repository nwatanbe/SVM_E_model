#! /usr/bin/env python
# -*- coding: utf-8 -*-
import ml_classification as ml_c
import time

a = time.time()
train_data_dir="../data/train/"
test_data_dir="../data/test/"
result_dir="../result/"

###Train
###Prepare train data
#Normalize
norm_dir=train_data_dir+"normalize/"
norm_out=norm_dir+"norm_max_min_-1_1.txt"

train_names,train_vecs,train_labels=ml_c.prepare_data_train(train_data_dir,norm_out)

###Optimize model
#Grid-search
c_list=[int(2**v) for v in range(0,6)]
gamma_list=[float(2**-v) for v in range(1,7)]
parameters={"C":c_list,"gamma":gamma_list}
cv=5
grid_cv_dir=result_dir+"model/optimize/"
output=grid_cv_dir+"grid_parameter.txt"
	
c_gamma_list=ml_c.grid_cv_svc(train_vecs,train_labels,cv,parameters,output)

###Cross validation and build model
#All Cross validation results
cv_out1=grid_cv_dir+"svc_"+str(cv)+"cv_f_score_results.tsv"
with open(cv_out1,"w") as g:g.write("C\tgamma\tF_score\n")

model_dir=result_dir+"model/"

for param in c_gamma_list:
	c=param[0];gamma=param[1]
	#Each Cross validation result
	cv_out2=grid_cv_dir+"svc_cv"+str(cv)+"_result_"+str(c)+"_"+str(gamma)+".tsv"
	
	#Cross validation
	ml_c.cross_validation_svc(train_vecs,train_labels,cv,cv_out1,cv_out2,c,gamma)
	
	#Build model
	out_model=model_dir+"svc_model_"+str(c)+"_"+str(gamma)+".pkl"
	clf=ml_c.build_svc_model(train_vecs,train_labels,out_model,c,gamma)

###Test
#Prepare test data
test_file=test_data_dir+"sample_test.vec"
test_names,test_vecs=ml_c.prepare_data_test(test_file,norm_out)

#Load model and test
c_gamma=str(c_gamma_list[0][0])+"_"+str(c_gamma_list[0][1])
model=model_dir+"svc_model_"+c_gamma+".pkl"
output=result_dir+"/test/test_results_"+c_gamma+".txt"

ml_c.svm_test(test_names,test_vecs,model,output)

print(time.time()-a)