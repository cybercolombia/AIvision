import os
import pwd
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

def create_dirs():
    parent_dir = os.getcwd()
    root = 'data'
    path = os.path.join(parent_dir, root)
    path1 = os.path.join(path, 'train')
    path2 = os.path.join(path, 'test')
    path11 = os.path.join(path1, 'images')
    path12 = os.path.join(path1, 'masks')
    path21 = os.path.join(path2, 'images')
    path22 = os.path.join(path2, 'masks')

    pred = 'predimages'
    
    try:
        os.makedirs(path11)
        os.makedirs(path12)
        os.makedirs(path21)
        os.makedirs(path22)
        os.makedirs(pred)
    except Exception as e:
        print("Directories already created.")
    
    return path11, path12, path21, path22, pred



def random_transforms(img, msk):
    angle = random.randint(0,90)
    new_img = img.rotate(angle)
    new_msk = msk.rotate(angle)
    fliph = random.randint(0,1)
    flipv = random.randint(0,1)
    if fliph == 1:
        new_img = new_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        new_msk= new_msk.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flipv == 1:
        new_img = new_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        new_msk = new_msk.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return new_img, new_msk



#------------------------------------------------------
pathTI, pathTM, pathVI, pathVM, preddir = create_dirs()

all_files = []
for child in Path('../images/').iterdir():
    all_files.append(child.name)

filenames = []
for child in Path('../labels/').iterdir():
    filenames.append(child.name)
#print(filenames)

predfiles = [f for f in all_files if f not in filenames] 
#print(predfiles)

fraction = 0.3
ntest = int(len(filenames) * fraction) 
random.seed(123)
testl = random.sample(filenames, ntest)
trainl = [i for i in filenames if i not in testl]
#print(testl, trainl)

n_transforms = 5
source_path1 = '../images/'
source_path2 = '../labels/'

for filename in predfiles:
    img = Image.open(source_path1+filename)
    img.save(preddir+"/"+filename)
    img.close()


for filename in trainl:
    img = Image.open(source_path1+filename)
    msk = Image.open(source_path2+filename)
    
    img.save(pathTI+"/"+filename[:-4]+"_0"+".png")
    msk.save(pathTM+"/"+filename[:-4]+"_0"+".png")
    for i in range(1,n_transforms+1):
        t_img, t_msk = random_transforms(img, msk)
        t_img.save(pathTI+"/"+filename[:-4]+"_"+str(i)+".png")
        t_msk.save(pathTM+"/"+filename[:-4]+"_"+str(i)+".png")
    #t_img.show()
    img.close()
    msk.close()

for filename in testl:
    img = Image.open(source_path1+filename)
    msk = Image.open(source_path2+filename)
    
    img.save(pathVI+"/"+filename[:-4]+"_0"+".png")
    msk.save(pathVM+"/"+filename[:-4]+"_0"+".png")
    for i in range(1,n_transforms+1):
        t_img, t_msk = random_transforms(img, msk)
        t_img.save(pathVI+"/"+filename[:-4]+"_"+str(i)+".png")
        t_msk.save(pathVM+"/"+filename[:-4]+"_"+str(i)+".png")
    #t_img.show()
    img.close()
    msk.close()
