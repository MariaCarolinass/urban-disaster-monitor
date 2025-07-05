from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil

# Caminhos das imagens e anotações
img_dir = 'dataset/train/images'
lbl_dir = 'dataset/train/labels'

img_files = sorted(glob(f'{img_dir}/*.jpg'))
lbl_files = sorted(glob(f'{lbl_dir}/*.txt'))

# Separar o conjunto em treino, validação e teste (70/15/15)
img_train, img_temp, lbl_train, lbl_temp = train_test_split(img_files, lbl_files, test_size=0.3, random_state=42)
img_val, img_test, lbl_val, lbl_test = train_test_split(img_temp, lbl_temp, test_size=0.5, random_state=42)

# Função segura para copiar arquivos
def organizar(dset, imgs, lbls):
    os.makedirs(f'dataset/{dset}/images', exist_ok=True)
    os.makedirs(f'dataset/{dset}/labels', exist_ok=True)

    for img, lbl in zip(imgs, lbls):
        img_dst = f'dataset/{dset}/images/{os.path.basename(img)}'
        lbl_dst = f'dataset/{dset}/labels/{os.path.basename(lbl)}'

        if img != img_dst:
            shutil.copy(img, img_dst)
        if lbl != lbl_dst:
            shutil.copy(lbl, lbl_dst)

# Apenas organizar val e test
organizar('val', img_val, lbl_val)
organizar('test', img_test, lbl_test)