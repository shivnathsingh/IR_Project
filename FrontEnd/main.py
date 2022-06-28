
from importlib.machinery import all_suffixes
from flask import Flask,redirect,request,url_for,render_template
from graphviz import render
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
from itertools import combinations
from nltk.corpus import wordnet 
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from gensim.parsing.preprocessing import remove_stopwords
from numpy import dot,mean
from numpy.linalg import norm
from sklearn.model_selection import train_test_split, cross_val_score
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
import joblib
import pickle
import math
from collections import Counter
import operator

app = Flask(__name__)

@app.route("/")
def begin():
    return render_template("index.html")

@app.route("/backtoHome")
def backtoHome():
    return redirect(url_for("begin"))

#function to get synonym of a given term
def syno(token):
  synonym = []
  website = 'https://www.synonym.com/synonyms/'
  myLinkFinder = website + token
  synoPage = requests.get(myLinkFinder,verify=False)
  currContent = BeautifulSoup(synoPage.content,"html.parser")
  html_Div = currContent.find('div', class_='section-list-wrapper show')
  row = html_Div.find_all('li')
  for i in row:
    synonym.append(i.get_text())
  for i in wordnet.synsets(token):
    synonym+=i.lemma_names()
  return set(synonym)

#function to lemmetize the word
def lemme(txt):
  txt=nltk.word_tokenize(txt)
  temp=[]
  for i in txt:
    temp.append(WordNetLemmatizer().lemmatize(i))
  return temp

#function to calculate jaccard coefficient of 2 set
def getJaccard(setA,setB):
  #print(setA,setB)
  return(len(setA.intersection(setB))/len(setA))


#function to calculate jaccard coefficient of 2 set
def getJaccard2(setA,setB):
  #print(setA,setB)
  return(len(setA.intersection(setB))/len(setA.union(setB)))



#fucntion to calculate cosine similarity between 2 lists
def cos_sim(listA, listB):
  cos_sim = dot(listA, listB) / (norm(listA) * norm(listB))
  return cos_sim

#function to process the input symptoms
def refine_symptoms(input_symptoms):
  input_symptoms=input_symptoms.lower()
  input_symptoms=input_symptoms.split(",")
  refined_symptoms=[]
  for i in input_symptoms:
    i=i.replace("-"," ")
    i=i.replace("'","")
    refined_symptoms.append(i)
  print(refined_symptoms)
  return refined_symptoms

#function to add synonyms of the symptoms in the list and extend the symptom list
def extend_symptoms(refined_symptoms):
  extended_symptom=[]
  for sym in refined_symptoms:
  #remove stopwords if any before taking its synonym
    sym=remove_stopwords(sym)
    sym = sym.split()
    set_sym = set()
    for comb in range(1, len(sym)+1):
        for sub_set in combinations(sym, comb):
            sub_set=' '.join(sub_set)
            sub_set = syno(sub_set) 
            set_sym.update(sub_set)
    extended_symptom.append(' '.join(set_sym))
  print(extended_symptom)
  return extended_symptom

#selecting relevent symptoms from the dataset
def relevent_symptom(data_symptom,extended_symptom):
  our_symptom=[]
  for sym in data_symptom:
    data_sym_tokens=sym.split()
    for user_sym in extended_symptom:
      if getJaccard(set(data_sym_tokens),set(user_sym.split()))>0.5:
        our_symptom.append(sym)
  print(our_symptom)
  return our_symptom

#confirming from user , the symptoms present in dataset
# def confirm_relevent_symptom(our_symptom):
#   poss_disease=set()
#   input_list=input("\n just confirm the relevent symptoms by entering their indices separated by comma\n").split(",")
#   for i in input_list:
#     new_symptom.append(our_symptom[int(i)-1])
#     poss_disease.update(list(dataset2[dataset2[our_symptom[int(i)-1]]==1]['label_disease']))
#   return new_symptom,poss_disease

#selecting co-occuring symptoms from the dataset
# def getCoOccuringSymptoms(new_symptom):
#   co_occur_symptom=[]
#   for disease in poss_disease:
#     temp_values = dataset2.loc[dataset2['label_disease'] == disease].values
#     temp_list=list(temp_values[0])
#     del temp_list[0]
#     index=0
#     for i in temp_list:
#       if i==1 and data_symptom[index] not in new_symptom:
#         co_occur_symptom.append(data_symptom[index])  
#       index+=1
#   return co_occur_symptom

#gives most probable diseases
def giveBestDiseases(predicted_list):
  bestK = predicted_list[0].argsort()
  bestK=bestK[-7:]
  bestK=bestK[::-1]
  return bestK

#get probabilities of the selected diseases
def get_probability(disease,new_symptom,bestK,disease_prob,data_symptom,dataset2):
  index=1
  for i in bestK:
    value_list=list(dataset2.loc[dataset2["label_disease"]==disease[i]].values)
    value_list=list(value_list[0])
    del value_list[0]
    ind=0
    dis_set=set()
    for j in value_list:
      if j==1:
        dis_set.add(data_symptom[ind])
      ind+=1
    pod=(len(dis_set.intersection(set(new_symptom)))+1)/(len(new_symptom)+1)
    disease_prob[i]=pod*91-(index*2)
    index+=1
  return disease_prob

def addMedical_history(sample_m_h,dataset2,data_symptom,disease_order):
  #value_list=[]
  for i in sample_m_h:
    print(i)
    #print(dataset2["asthma"])
    value_list=list(dataset2.loc[dataset2["label_disease"]==i].values)
    #print(value_list)
    value_list=list(value_list[0])
    del value_list[0]
    idx=0
    mh_list=[]
    for value in value_list:
      if(value==1):
        mh_list.append(data_symptom[idx])
      idx+=1
    #print(mh_list)
    for j in disease_order:
      temp_list=list(dataset2.loc[dataset2["label_disease"]==j].values)
      temp_list=list(temp_list[0])
      del temp_list[0]
      idx=0
      curr_list=[]
      for value in temp_list:
        if(value==1):
          curr_list.append(data_symptom[idx])
        idx+=1
      #print(curr_list)
      print(j,"->",getJaccard(set(curr_list),set(mh_list)))
      if(getJaccard(set(curr_list),set(mh_list))>=0.5):
          #ranking up disease by scaling the difference of jaccard value and 0.5 by 30
        disease_order[j]+=abs(0.5-getJaccard(set(curr_list),set(mh_list)))*30
  return disease_order



@app.route('/predict',methods=['POST'])
def predict():

    

    all_Features = [ele for ele in request.form.values()]
    userName = all_Features[0]
    all_Symptoms = all_Features[1]
    medical_His = []
    if(all_Features[2]=="none"):
        x=2
    else:
        arr=all_Features[2].split(",")
        for i in arr:
            medical_His.append(i)
        print("medical history"+str(medical_His))


    dataset=pd.read_csv("dataset//df_final_comb.csv")
    dataset2=pd.read_csv("dataset//df_final_normal (2).csv")
    data_symptom=list(dataset.columns)
    data_symptom.remove('label_disease')
    data_symptom
    x_of_data = dataset2.iloc[:, 1:]
    y_of_data = dataset2.iloc[:, 0:1]

    refined_symptoms=refine_symptoms(all_Symptoms)
    extended_symptom = []
    extended_symptom = extend_symptoms(refined_symptoms)
    our_symptom=[]
    our_symptom= relevent_symptom(data_symptom,extended_symptom)
    # new_symptom=[]
    # new_symptom,poss_disease=confirm_relevent_symptom(our_symptom)
    # co_occur_symptom=[]
    # co_occur_symptom=getCoOccuringSymptoms(our_symptom) 
    # dict_symp = dict(Counter(co_occur_symptom))
    # dict_symp_sorted = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)
    # our_symptom.append(dict_symp_sorted[0][0])
    # our_symptom.append(dict_symp_sorted[1][0])
    my_sample=[]
    for x in range(0,len(data_symptom)):
      my_sample.append(0)
    for val in our_symptom:
      my_sample[data_symptom.index(val)]=1
    #model = pickle.load(open("/content/log_reg (1).pkl", "rb"))
    model = joblib.load("dataset//log_reg (1).pkl")
    predicted_list = model.predict_proba([my_sample])
    bestK=giveBestDiseases(predicted_list)
    disease = list(y_of_data['label_disease'])
    disease.sort()
    disease_prob={}
    dis_set=set()
    disease_prob=get_probability(disease,our_symptom,bestK,disease_prob,data_symptom,dataset2)

    dict_final_sorted = sorted(disease_prob.items(), key=operator.itemgetter(1),reverse=True)
    ind=0
    old=0
    disease_order={}
    res=""
    for i in dict_final_sorted:
      if(old==disease_prob[i[0]]):
        #print(ind,disease[i[0]],disease_prob[i[0]],"%")
        res+=str(disease[i[0]])+" "+str(disease_prob[i[0]])+"%\n"
      else:
        #print(ind,disease[i[0]],disease_prob[i[0]],"%")
        res+=str(disease[i[0]])+" "+str(disease_prob[i[0]])+"%\n"
      disease_order[disease[i[0]]]=disease_prob[i[0]]
      ind+=1
      old=disease_prob[i[0]]
    disease_order = addMedical_history(medical_His, dataset2, data_symptom, disease_order)
    dict_order = sorted(disease_order.items(), key=operator.itemgetter(1), reverse=True)
    myDiseases =[]
    for i in dict_order:
        # print(ind,i[0],i[1],"%")
        myDiseases.append(str(i[0])+" "+str(i[1]) + "%")
    dis_0 = myDiseases[0]
    dis_1 = myDiseases[1]
    dis_2 = myDiseases[2]
    dis_3 = myDiseases[3]
    dis_4 = myDiseases[4]
    dis_5 = myDiseases[5]
    dis_6 = myDiseases[6]
    
    return render_template('result.html',name = userName,bimamri_0=dis_0,bimamri_1=dis_1,bimamri_2=dis_2,
    bimamri_3=dis_3,bimamri_4=dis_4,bimamri_5=dis_5 ,bimamri_6=dis_6,
    medicalHistory = all_Features[2])




if __name__ == "__main__": 
    app.run()
