#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os,glob,joblib,random
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.svm import SVC

pit=12345
random.seed(pit)

def make_dirs(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def prepare_data_train(train_data_dir,norm_out):
	train_data=[]
	posi_files=glob.glob(train_data_dir+"*_positive*")
	for posi_file in posi_files:
		with open(posi_file) as f:
			samples=f.read().split("\n")
			
		for sample in samples:
			train_data.append("1,,"+sample)
		
	nega_files=glob.glob(train_data_dir+"*_negative*")
	for nega_file in nega_files:
		with open(nega_file) as f:
			samples=f.read().split("\n")
			if samples[-1]=="":del samples[-1]
			
		for sample in samples:
			train_data.append("0,,"+sample)
	
	random.shuffle(train_data)
	
	train_names=[data.split(",,")[1] for data in train_data]
	train_vecs=[list(map(float,data.split(",,")[2].split("\t"))) for data in train_data]
	train_labels=[int(data.split(",,")[0]) for data in train_data]
	
	make_dirs("/".join(norm_out.split("/")[:-1])+"/")
	train_vecs=train_seq_normalization(train_vecs,norm_out)
	
	return train_names,train_vecs,train_labels

def train_seq_normalization(train_vecs,norm_out):
	with open(norm_out,"w") as g:
		train_vecs=np.array(train_vecs)
		for h in range(train_vecs.shape[1]):
			Max=np.max(train_vecs[:, h])
			Min=np.min(train_vecs[:, h])
			for i in range(len(train_vecs)):
				if Max == Min:
					train_vecs[i , h]= 0
				else:
					train_vecs[i , h]= (2*train_vecs[i , h]-Max-Min)/(Max-Min)
			g.write(str(Max)+"\t"+str(Min)+"\n")
		
	return train_vecs

def grid_cv_svc(train_vecs,train_labels,cv,parameters,output):
	svc=SVC(kernel='rbf',class_weight='balanced', probability=True)
	clf=GridSearchCV(estimator=svc, param_grid=parameters,cv=cv,scoring=('f1'),n_jobs=cv)
	clf.fit(train_vecs, train_labels)
	
	make_dirs("/".join(output.split("/")[:-1])+"/")
	with open(output,"w") as g:
		g.write("params\tf_score_mean\n")
		for x,y in zip(clf.cv_results_["params"],clf.cv_results_["mean_test_score"]):
			g.write(str(x)+"\t"+str(y)+"\n")
	
	values=[[x.get("C"),x.get("gamma"),y] for x,y in zip(clf.cv_results_["params"],clf.cv_results_["mean_test_score"])]
	table=pd.DataFrame(np.array(values), columns=["C","gamma","fscore"])
	table=table.sort_values("fscore",ascending=False)[:5]
	
	c_gamma_list=[[x,y] for x,y in zip(table["C"],table["gamma"])]
	
	return c_gamma_list
	
def cross_validation_svc(train_vecs,train_labels,cv,cv_out1,cv_out2,c,gamma):
	clf=SVC(C=c,kernel="rbf", gamma=gamma,class_weight='balanced', probability=True)
	cv_results=cross_validate(clf,train_vecs,train_labels,cv=cv,
	scoring=('f1','precision','recall','roc_auc','accuracy'),n_jobs=cv)
	
	with open(cv_out1,"a") as g:
		precision=sum(cv_results["test_precision"].tolist())/len(cv_results["test_precision"].tolist())
		recall=sum(cv_results["test_recall"].tolist())/len(cv_results["test_recall"].tolist())
		f_score=2.0*precision*recall/(precision+recall)
		try:g.write(str(c)+"\t"+str(gamma)+"\t"+str(f_score)+"\n")
		except ZeroDivisionError:g.write(str(c)+"\t"+str(gamma)+"\tZeroDivisionError\n")
		
	with open(cv_out2,"w") as h:
		h.write("(c,gamma)=("+str(c)+","+str(gamma)+")\n")
		h.write("value\t")
		for x in range(1,cv+1):h.write("cv"+str(x)+"\t")
		h.write("\n")
		h.write("accuracy\t"+"\t".join(map(str,cv_results["test_accuracy"].tolist()))+"\n"+
		"precision\t"+"\t".join(map(str,cv_results["test_precision"].tolist()))+"\n"+
		"recall\t"+"\t".join(map(str,cv_results["test_recall"].tolist()))+"\n"+
		"f_score\t"+"\t".join(map(str,cv_results["test_f1"].tolist()))+"\n"+
		"roc_auc\t"+"\t".join(map(str,cv_results["test_roc_auc"].tolist()))+"\n")

def build_svc_model(train_vecs,train_labels,out_model,c,gamma):
	clf= SVC(C=c,kernel='rbf',gamma=gamma,class_weight="balanced", probability=True)
	clf.fit(train_vecs,train_labels)
	joblib.dump(clf,out_model)
	return clf

#Test
def prepare_data_test(test_file,norm_out):
	with open(test_file) as f:
		samples=f.read().split("\n")
		if samples[-1]=="":del samples[-1]
	
	test_names=[data.split(",,")[0] for data in samples]
	test_vecs=[list(map(float,data.split(",,")[1].split("\t"))) for data in samples]
	test_vecs=test_seq_normalization(test_vecs,norm_out)
	
	return test_names,test_vecs

def test_seq_normalization(test_vecs,norm_out):
	with open(norm_out) as f:
		lines=f.read().split("\n")
		if lines[-1]=="":del lines[-1]
	
	max_list=[float(line.split("\t")[0]) for line in lines]
	min_list=[float(line.split("\t")[1]) for line in lines]
	test_vecs=np.array(test_vecs)
	for h in range(len(max_list)):
		Max=max_list[h]
		Min=min_list[h]
		for i in range(len(test_vecs)):
			if Max == Min:
				test_vecs[i , h]= 0
			else:
				test_vecs[i , h]= (2*test_vecs[i , h]-Max-Min)/(Max-Min)
		
	return test_vecs

def svm_test(test_names,test_vecs,model,output):
	make_dirs("/".join(output.split("/")[:-1]))
	clf=joblib.load(model)
	with open(output,"w") as g:
		g.write("name\tpred_label\tpositive_probability\tdistance\n")
		
		pred=clf.predict(test_vecs)
		proba_p=clf.predict_proba(test_vecs)[:,1]
		decision_=clf.decision_function(test_vecs)
		for x in range(len(test_names)):
			g.write(test_names[x]+"\t"+str(pred[x])+"\t"+str(proba_p[x])+"\t"+str(decision_[x])+"\n")

