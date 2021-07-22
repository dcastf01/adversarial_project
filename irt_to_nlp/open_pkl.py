import pickle


with open('/home/dcast/adversarial_project/irt_to_nlp/AAAI_Results/WSBias/SST-2/sst2-predictions-128iter.pkl', 'rb') as f:
    data = pickle.load(f)
    
    print(data)
    