import os
import sys
sys.path.append("/home/dcast/adversarial_project")
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from art.attacks.evasion import ElasticNet
from art.estimators.classification import PyTorchClassifier
from openml.config import CONFIG, Dataset
from openml.datamodule import OpenMLDataModule
from PIL import Image
from pytorch_lightning.core import datamodule
from tqdm import tqdm


#cd /home/dcast/adversarial_project ; /usr/bin/env /home/dcast/anaconda3/envs/deep_learning_torch/bin/python -- /home/dcast/adversarial_project/openml/generate_adversarial_image_to_4_application.py 
#cd /home/dcast/adversarial_project ; /usr/bin/env /home/dcast/anaconda3/envs/deep_learning_torch/bin/python  -- /home/dcast/adversarial_project/openml/adversarial_images/generate_adversarial_image_to_4_application.py 
def get_image_label_diff_index(dataset):
    images=[]
    diffs=[]
    labels=[]
    indexs=[]

    for i in range(len(dataset)):
        # a=dataset_train[i]
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
                                            input_size=32)
data_module.setup()

dataloader_train=data_module.train_dataloader()
dataset_train=dataloader_train.dataset
print("Generating datasets train")
x_train,y_train,diff_train,indexs_train=get_image_label_diff_index(dataset_train)

dataloader_test=data_module.val_dataloader()
dataset_test=dataloader_test.dataset
print("Generating datasets test")
x_test,y_test,diff_test,indexs_test=get_image_label_diff_index(dataset_test)

def create_classifier_art():
    in_chans=1
    extras=dict(in_chans=in_chans)
    model=timm.create_model("resnet50",
                            pretrained=True,
                            num_classes=10,
                            **extras
                            )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(in_chans, 32, 32),
        nb_classes=10,
        )
    
    return classifier
classifier=create_classifier_art()

print("Training the classifier")
classifier.fit(x_train,y_train,batch_size=128,nb_epochs=10)
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

def create_adversarial_image(image_original,classifier,y,lr=0.00001):
    def calculate_l2(batch_original,batch_adversarial):
        return np.linalg.norm(batch_original-batch_adversarial)
    accuracy=1
    loop=0
    image_to_modified=image_original
    
    prediction_initial = classifier.predict(image_to_modified)
    result=np.argmax(prediction_initial, axis=1)
    accuracy_initial = np.sum(np.argmax(prediction_initial, axis=1) == np.argmax(y, axis=1)) / len(y)
    if accuracy_initial!=1:
        was_modified=False
        lminimumsquare=None
        return image_to_modified,accuracy_initial,lminimumsquare,was_modified
    while accuracy==1: #puede entrar en bucle por lo que arreglar
        # Step 6: Generate adversarial test examples
        was_modified=True
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
        # linfinite = calculate_linifinite(image_original,x_test_adv,dim)
        # lzero=calculate_l0(image_original,x_test_adv,dim)
        lminimumsquare=calculate_l2(image_original,x_test_adv)
        
        
        print("Accuracy on adversarial test batch: {}%".format(accuracy * 100))
        # print("perturbation linfinite: {}%".format(linfinite))
        # print("perturbation L0: {}".format(lzero))
        print("perturbation L2: {}".format(lminimumsquare))
        
        
        loop+=1
        # image_np=x_test_adv[0,...].squeeze()
        # im = Image.fromarray(np.uint8(image_np*255))
        # im.save("delete001.png")
    return x_test_adv,accuracy,lminimumsquare,was_modified

indices=[]
lminimumsquares=[]
name_csv=f"{dataset_enum.name}_data_adversarial.csv"
path_base="/home/dcast/adversarial_project/openml/adversarial_images"
path_file=os.path.join(path_base,name_csv)
# path_file2=os.path.join(path_base,"algo"+name_csv)
pre_df=pd.read_csv(path_file)
number_iterations=[]
difficulties=[]
clases=[]
is_adversariales=[]
lrs=[]
images_flat=[]
i=0
lr=1e-7
num_images=350
for img,y,index,diff in tqdm(zip(x_test,y_test,indexs_test,diff_test),total=num_images):
    if index not in pre_df.id:
        img=np.expand_dims(img,axis=0)
        y=np.expand_dims(y,axis=0)
        x_test_adv,accuracy,lminimumsquare,was_modified=create_adversarial_image(img,classifier,y,lr)
        if not was_modified:
            continue
        if accuracy==0:
            is_adversarial=True
        else:
            is_adversarial=False
        img_flat=x_test_adv.reshape(-1)
        images_flat.append(img_flat)
        # einum_reshape='ijk->jki'
        # not_img_flat=img_flat.reshape((3,32,32))  
        # img_recreated=np.einsum(einum_reshape, not_img_flat)
        # image_np=x_test_adv[0,...].squeeze()
        # im = Image.fromarray(np.uint8(img_recreated*255))
        # im.save("delete001.png")
        indices.append(index)
        # linifinites.append(float(linfinite))
        # lzeros.append(float(lzero))
        lminimumsquares.append(float(lminimumsquare))
        # perturbations.append(float(pert))
        difficulties.append(float(diff))
        clases.append(np.argmax(y))
        is_adversariales.append(is_adversarial)
        lrs.append(lr)
        
        #solo haga i imagenes
        i+=1
        if i>=num_images:
            break    
print("creating dataframes")
df_image=pd.DataFrame(images_flat).add_prefix("pixel")
# print(df_image.head())
data={  "id":indices,
        
        "l2":lminimumsquares,

        "diff":difficulties,
        "class":clases,
        "adversarial":is_adversariales,
        "lr":lrs,
        # "number_iterations_necessary":number_iterations
              }
#el fallo se encuentra que pre_df las columnas se leencomo string por tanto está como'3071' y en df son 3071
df1=pd.DataFrame(data=data)
print(df1.head())
print(df_image)
df=pd.concat([df1,df_image],axis=1)
# df1=df1.merge(image)
print(df.head())
# df.to_csv(path_file,index=False)
df_total=pd.concat([df,pre_df])
df_total.to_csv(path_file,index=False)
# print(df_total.head())
print("finish")
