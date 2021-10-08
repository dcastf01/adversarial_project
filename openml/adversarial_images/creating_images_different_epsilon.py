import os
import sys
sys.path.append("/home/dcast/adversarial_project")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.attacks.evasion import (CarliniL2Method, ElasticNet,
                                 FeatureAdversariesPyTorch,ProjectedGradientDescent)
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from PIL import Image
from pytorch_lightning.core import datamodule
from tqdm import tqdm
from openml.config import CONFIG, Dataset
from openml.datamodule import OpenMLDataModule
from enum import Enum
import cv2
class Attacks(Enum):
    carlini=0
    elasticnet=1
    projected_gradient_descent=2
    
class TargetAttack(Enum):
    l0="L0"
    l1="L1"
    l2="L2" 
    linfinite="EN"
    notarget="no_target"

#cd /home/dcast/adversarial_project ; /usr/bin/env /home/dcast/anaconda3/envs/deep_learning_torch/bin/python -- /home/dcast/adversarial_project/openml/creating_images_different_epsilon.py 

path_to_save_img="/home/dcast/adversarial_project/openml/adversarial_images"
    
def save_img(img,label_one_hot,extra="none"):
    label=np.argmax(label_one_hot)
    img=img[0][0]*255
    first_array=img
    # first_array=first_array.numpy()
    #Not sure you even have to do that if you just want to visualize it
    # first_array=255*first_array
    # first_array=self._watermark_by_class(first_array)
    first_array=first_array.astype("uint8")
    if extra:
        path_fn=os.path.join(path_to_save_img,extra+" "+
                            str(label)+".png")
    else:
        path_fn=os.path.join(path_to_save_img,
                            path_to_save_img.split("/")[-1]+str(label)+".png")
    first_array=first_array.astype("uint8")
    cv2.imwrite(path_fn,first_array)
    plt.imshow(first_array)
    #Actually displaying the plot if you are not in interactive mode
    plt.show()
    #Saving plot
    plt.savefig(os.path.join(path_to_save_img,
                                "save.png"))

def get_image_label_diff_index(dataset):
    images=[]
    diffs=[]
    labels=[]
    indexs=[]

    for i in range(len(dataset)):
        img,target,index,label=dataset[i]

        label=torch.nn.functional.one_hot(label,num_classes=10)
        label=torch.squeeze(label,dim=0)
        images.append(img.detach().numpy())
        diffs.append(target.detach().numpy())
        labels.append(label.detach().numpy())
        indexs.append(index)
        
        
    x=np.stack(images, axis=0)
    x=(x+1)/2
    # print(np.amax(x))
    # print(np.amin(x))
    y=np.stack(labels,axis=0)
    diffs=np.stack(diffs,axis=0)
    indexs=np.stack(indexs,axis=0)
    return x,y,diffs,indexs

dataset_enum=Dataset.mnist784_ref
batch_size=64
workers=0
path_data_csv=CONFIG.path_data

# Step 1: Load the MNIST dataset
data_module=OpenMLDataModule(data_dir=os.path.join(path_data_csv,dataset_enum.value),
                                            batch_size=batch_size,
                                            dataset=dataset_enum,
                                            num_workers=workers,
                                            pin_memory=True,
                                            input_size=28)
data_module.setup()

dataloader_train=data_module.train_dataloader()
dataset_train=dataloader_train.dataset


x_train,y_train,diff_train,indexs_train=get_image_label_diff_index(dataset_train)

dataloader_test=data_module.val_dataloader()
dataset_test=dataloader_test.dataset

x_test,y_test,diff_test,indexs_test=get_image_label_diff_index(dataset_test)
# Step 2: Create the model

model = nn.Sequential(
    nn.Conv2d(1, 4, 5), nn.ReLU(), nn.MaxPool2d(2, 2), 
    nn.Conv2d(4, 10, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Flatten(), 
    nn.Linear(4*4*10, 100),    
    nn.Linear(100, 10)
)

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

classifier.fit(x_train,y_train,batch_size=128,nb_epochs=5)

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


def calculate_l0(batch_original,batch_adversarial,dim):
    # image_original==x_test_adv
    matrix_bool=batch_original==batch_adversarial
    inverse_matrix= np.logical_not(matrix_bool)
    l0=np.count_nonzero(inverse_matrix, axis=dim)
    return l0

def calculate_l2(batch_original,batch_adversarial):
    return np.linalg.norm(batch_original-batch_adversarial)

def calculate_linifinite(batch_original,batch_adversarial,dim):
    return np.mean(np.amax(np.abs(batch_original - batch_adversarial), axis=dim))
    


def create_adversarial_image(image_original,y,attack,target,lr=0.000001,pert=0.1):
    def get_attack(attack,target):
    
        if attack==Attacks.elasticnet:
            # lr=pert
            attack = ElasticNet(
                    classifier,
                    targeted=False,
                    decision_rule=target,
                    batch_size=1,
                    learning_rate=lr,
                    max_iter=100, # 1000 recomendado por Iveta y Stefan
                    binary_search_steps=25, # 50 recomendado por Iveta y Stefan
                    # layer=7,
                    # delta=35/255,
                    # optimizer=None,
                    # step_size=1/255,
                    # max_iter=100,
                )
        elif attack==Attacks.projected_gradient_descent:
            if target==TargetAttack.notarget.value:
                attack=ProjectedGradientDescent(
                    classifier,
                    eps=pert,
                    eps_step=0.05
                    
                    
                )
                target="no_target"
            else:
                raise Exception("set no target if you use Projected Attack")
        
        return attack
    
    attack_fn=get_attack(attack,target)
    accuracy=1
    loop=0
    image_to_modified=image_original
    
    while accuracy==1: #puede entrar en bucle por lo que arreglar
        # Step 6: Generate adversarial test examples
        if loop>0:
            image_to_modified=x_test_adv
        
        x_test_adv = attack_fn.generate(image_to_modified)
        # save_img(x_test_adv,y,extra="probando")
        # Step 7: Evaluate the ART classifier on adversarial test examples

        predictions = classifier.predict(x_test_adv)
        result=np.argmax(predictions, axis=1)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)

        dim = tuple(range(1, len(image_original .shape)))
        # a=np.abs(image_original - x_test_adv)
        
        # b=np.amax(np.abs(image_original - x_test_adv), axis=dim)
        # # b=np.amax(a, axis=dim)
        # c=np.mean(b)
        # testing=np.abs(image_original-image_original)
        # print("testing",testing)
        linfinite = calculate_linifinite(image_original,x_test_adv,dim)
        lzero=calculate_l0(image_original,x_test_adv,dim)
        lminimumsquare=calculate_l2(image_original,x_test_adv)
        
        
        print("Accuracy on adversarial test batch: {}%".format(accuracy * 100))
        print("perturbation linfinite: {}%".format(linfinite))
        print("perturbation L0: {}".format(lzero))
        print("perturbation L2: {}".format(lminimumsquare))
        
        
        loop+=1
        # image_np=x_test_adv[0,...].squeeze()
        # im = Image.fromarray(np.uint8(image_np*255))
        # im.save("delete001.png")
        
    return accuracy,linfinite,lzero,lminimumsquare,loop

i=0
# create_adversarial_image(x_test,y_test)

indices=[]
linifinites=[]
lzeros=[]
lminimumsquares=[]
# perturbations=[]
difficulties=[]
clases=[]
is_adversariales=[]
lrs=[]
perts=[]
targets=[]
attacks=[]

target=TargetAttack.l1
target=target.value
attack=Attacks.elasticnet
name_csv=f"{attack.name}_{target}_{dataset_enum.name}_data.csv"
# pre_df=pd.read_csv(name_csv)
pre_df=pd.DataFrame()
number_iterations=[]


# attack_fn=get_attack(target)
lr=0.000001
pert=0.001
num_images=2000
for img,y,index,diff in tqdm(zip(x_test,y_test,indexs_test,diff_test),total=num_images):
    # if index not in pre_df.id:
        img=np.expand_dims(img,axis=0)
        y=np.expand_dims(y,axis=0)
        accuracy,linfinite,lzero,lminimumsquare,number_iteration=create_adversarial_image(img,y,attack,target,lr,pert)
        if accuracy==0:
            is_adversarial=True
        else:
            is_adversarial=False
        indices.append(index)
        linifinites.append(float(linfinite))
        lzeros.append(float(lzero))
        lminimumsquares.append(float(lminimumsquare))
        # perturbations.append(float(pert))
        difficulties.append(float(diff))
        clases.append(np.argmax(y))
        is_adversariales.append(is_adversarial)
        lrs.append(lr)
        perts.append(pert)
        number_iterations.append(number_iteration)
        attacks.append(attack.name)
        targets.append(target)
        
        #solo haga i imagenes
        i+=1
        if i>num_images:
            break
    
data={"id":indices,
              "linfinite":linifinites,
                "l2":lminimumsquares,
               "l0" :lzeros,
              "diff":difficulties,
              "class":clases,
              "adversarial":is_adversariales,
              "lr":lrs,
              "pert":pert,
              "number_iterations_necessary":number_iterations,
              "attack":attacks,
              "target":targets
              }
df1=pd.DataFrame(data=data)
df_total=pd.concat([df1,pre_df])
path_to_save="/home/dcast/adversarial_project/openml/adversarial_images"
df_total.to_csv(os.path.join(path_to_save,name_csv),index=False)
# Step 8: Inspect results
print("finish")
# # orig 7, guide 6
# image_np=x_test_adv[0,...].squeeze()
# im = Image.fromarray(np.uint8(image_np*255))
# im.save("delete001.png")
# plt.show()
