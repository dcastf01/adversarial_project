import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.attacks.evasion import (CarliniL2Method, ElasticNet,
                                 FeatureAdversariesPyTorch)
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from PIL import Image
from pytorch_lightning.core import datamodule
from tqdm import tqdm
from openml.config import CONFIG, Dataset
from openml.datamodule import OpenMLDataModule

#cd /home/dcast/adversarial_project ; /usr/bin/env /home/dcast/anaconda3/envs/deep_learning_torch/bin/python -- /home/dcast/adversarial_project/openml/creating_images_different_epsilon.py 
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

# Step 1: Load the MNIST dataset

# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# # Step 1a: Swap axes to PyTorch's NCHW format

# x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
# x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

dataset_enum=Dataset.fashionmnist_ref
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

dataloader_test=data_module.test_dataloader()
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




# Step 5: prepare a batch of source and guide images
# valid = np.argmax(y_test, axis=1)[:100] != np.argmax(y_test, axis=1)[100:200]
# source = x_test[:100][valid][:32]
# guide = x_test[100:200][valid][:32]

# # subset_to_transform_in_adversarial=x_test[0:10]
# # y_false1=y_test[0:10]
# # y_false=np.array([[1,0,0,0,0,0,0,0,0,0],
# #                   [1,0,0,0,0,0,0,0,0,0]])


# # Step 6: Generate adversarial test examples
# attack = ElasticNet(
#             classifier,
#             targeted=False,
#             decision_rule="L2",
#             batch_size=1,
#             max_iter=125, # 1000 recomendado por Iveta y Stefan
#             binary_search_steps=25,

# )
# x_test_adv = attack.generate(source)

# # Step 7: Evaluate the ART classifier on adversarial test examples

# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:100][valid][:32], axis=1)) / len(y_test[:100][valid][:32])

# dim = tuple(range(1, len(source.shape)))
# a=np.abs(source - x_test_adv)
# b=(np.amax(np.abs(source - x_test_adv), axis=dim))

# pert = np.mean(np.amax(np.abs(source - x_test_adv), axis=dim))
# print("Accuracy on adversarial test batch: {}%".format(accuracy * 100))
# print("Average perturbation: {}%".format(pert))


# print(b)
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
    


def create_adversarial_image(image_original,y,lr=0.000001):
    accuracy=1
    loop=0
    image_to_modified=image_original
    
    while accuracy==1: #puede entrar en bucle por lo que arreglar
        # Step 6: Generate adversarial test examples
        if loop>0:
            image_to_modified=x_test_adv
        attack = ElasticNet(
            classifier,
            targeted=False,
            decision_rule="L2",
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
        x_test_adv = attack.generate(image_to_modified)

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

name_csv=f"{dataset_enum.name}_data.csv"
pre_df=pd.read_csv(name_csv)
# pre_df=pd.DataFrame()
number_iterations=[]

lr=1e-10
num_images=100
for img,y,index,diff in tqdm(zip(x_test,y_test,indexs_test,diff_test),total=num_images):
    if index not in pre_df.id:
        img=np.expand_dims(img,axis=0)
        y=np.expand_dims(y,axis=0)
        accuracy,linfinite,lzero,lminimumsquare,number_iteration=create_adversarial_image(img,y,lr)
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
        number_iterations.append(number_iteration)
        
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
              "number_iterations_necessary":number_iterations
              }
df1=pd.DataFrame(data=data)
df_total=pd.concat([df1,pre_df])
df_total.to_csv(name_csv,index=False)
# Step 8: Inspect results
print("finish")
# # orig 7, guide 6
# image_np=x_test_adv[0,...].squeeze()
# im = Image.fromarray(np.uint8(image_np*255))
# im.save("delete001.png")
# plt.show()
