# Huong dan chay tren Colab

## Cach 1: Dung Git (khuyen nghi)
1. Day code len GitHub (tren may ban):
   - git init
   - git add .
   - git commit -m "init"
   - git remote add origin <URL_REPO_GITHUB>
   - git push -u origin main

2. Tren Colab (chay tung cell):
   - !git clone <URL_REPO_GITHUB>
   - %cd kskss
   - !pip install -r requirements_colab.txt

3. Mount Google Drive de lay du lieu:
   - from google.colab import drive
   - drive.mount('/content/drive')

4. Tao lien ket den du lieu (neu du lieu nam trong Drive):
   - !ln -s /content/drive/MyDrive/<duong_dan_data> ./data
   - !ln -s /content/drive/MyDrive/<duong_dan_labels> ./labels

5. Chay train:
   - !python train.py

## Cach 2: Upload zip
1. Nen folder project thanh file .zip, upload len Colab.
2. Giai nen va chay:
   - !unzip <ten_file>.zip -d /content/kskss
   - %cd /content/kskss
   - !pip install -r requirements_colab.txt
   - !python train.py
