import numpy as np 
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub

 
# Uncomment this when running the code (dataset) for the first time
# import numpy as np
# import nltk
# import re
# import pandas as pd
# import pickle
# import tensorflow as tf
# import tensorflow_hub as hub
# dataset = pd.read_csv("data.csv")
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(module_url)
# def embed(input):
#   return model([input])
# dataset['Question_Vector'] = dataset.Questions.map(embed)
# dataset['Question_Vector'] = dataset.Question_Vector.map(np.array)
# pickle.dump(dataset, open('data.pkl', 'wb'))
  
        
class DialogueManager(object):
    def __init__(self):
 
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.dataset = pickle.load(open('dataset.pkl', mode='rb'))
        self.questions = self.dataset.Questions
        self.QUESTION_VECTORS = np.array(self.dataset.Question_Vector)
        self.COSINE_THRESHOLD = 0.5       
        
 
         
        
    def embed(self,input):
        return self.model([input])  
        
    def cosine_similarity(self,v1, v2):
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if (not mag1) or (not mag2):
            return 0
        return np.dot(v1, v2) / (mag1 * mag2)
        
        
    def semantic_search(self, query, data, vectors):        
        query_vec = np.array(self.embed(query))
        res = []
        for i, d in enumerate(data):
            qvec = vectors[i].ravel()
            sim = self.cosine_similarity(query_vec, qvec)
            res.append((sim, d[:100], i))
        return sorted(res, key=lambda x : x[0], reverse=True)        
            
    
    def generate_answer(self, question):
        '''This will return list of all questions according to their similarity,but we'll pick topmost/most relevant question'''
        most_relevant_row = self.semantic_search(question, self.questions, self.QUESTION_VECTORS)[0]
        print(most_relevant_row)
        if most_relevant_row[0][0]>=self.COSINE_THRESHOLD:
            answer = self.dataset.Answers[most_relevant_row[2]]
        else:
            # answer = self.chitchat_bot.get_response(question)
            answer = 'I\'m sorry I can\'t undestand your question.'
       return answer     
      
         
