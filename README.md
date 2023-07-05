![Untitled (1)](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/d3bca362-3b1d-4a6a-8982-6836656793f5)

## 📸 Handbone X-ray Semantic Segmentation Task
---

- 뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.
- Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.
### **📆** 대회 일정 : 2023.06.05 ~ 2023.06.22

---
### **🗂️** Dataset

---

- 전체 이미지 개수 : 1100장 (train 800장 + test 300장)
- class : 29개 (’finger-1~19’, ‘Trapezium’, ‘Trapezoid’, ‘Capitate’, ‘Hamate’, ‘Scaphoid’, ‘Lunate’, ‘Triquetrum’, ‘Pisiform’, ‘Radius’, ‘Ulna’)
- 이미지 크기 : (2048, 2048)
- annotation file : 각 class에 대한 points 정보
- meta data : ID, 나이, 성별, 체중, 키, 네일아트 유무

## 👨🏻‍💻 👩🏻‍💻 팀 구성

-------------
|![logo1](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/e8938309-329d-4cf3-8b6c-180df65e3894)|![logo2](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/95d4bbb7-645e-4fd6-b18c-9b5d16a26c82)|![logo3](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/a23de15c-8927-4207-ab8d-d78ff3a8971b)|![logo4](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/296eadab-cb6e-47f8-ae0f-55995476d4ba)|![logo5](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/c197b8b6-c898-4e2d-90ca-90700943e9ca)|
| :---: | :---: | :---: | :---: |  :---: |
| [김용우](https://github.com/yongwookim1) | [박종서](https://github.com/justinpark820) | [서영덕](https://github.com/SeoYoungDeok) |[신현준](https://github.com/june95) |[조수혜](https://github.com/suhyehye) |

## 📊 EDA 결과

---

- label name이 바뀌어 있는 annotation이 존재
- 악세서리가 뼈로 labeling되어 있는 경우가 존재
- 네일아트가 있거나 수술로 인한 철심이 있는 경우가 존재
- test dataset에서 정방향 사진보다 손목을 꺾어서 촬영한 사진이 많음(train set: 약 12% / test set: 약 60%)

  ## 🍀 Folder Structure

---

```
├── codebook : EDA, ensemble, visualize등의 코드를 작성
│   ├── ensemble.py
│   ├── metadata_EDA.ipynb
│   ├── metadata_to_csv.ipynb
│   ├── soft_ensemble.ipynb
│   └── visualize_inference.ipynb
├── mmsegmentation : mmsegmentation library baseline code
│   ├── configs 
│   ├── mmseg
│   ├── tests
│   ├── tools
│   └── setup.py
├── src
│   ├── architecture 
│   ├── config
│   ├── data
│   ├── loss.py
│   ├── model.py
│   ├── scheduler.py
│   ├── train.py
│   └── utils.py
├── poetry.toml
├── pyproject.toml
├── run.py
├── tuna.py
└── .gitignore
```

## 📕 Code book

---

- ensemble.py : 결과 csv를 사용해서 hard voting ensemble을 진행하는 파일
- metadata_EDA.ipynb : metadata 파일의 데이터들에 대한 EDA를 진행하는 파일
- metadata_to_csv.ipynb : metadata의 오류들을 수정하고 xlsx 형식을 csv파일로 변경 후 저장하는 파일
- soft_ensemble.ipynb : ensemble할 모델들을 불러와 sigmoid를 통과하기 전 결과들을 평균내어 그 값으로 inference하는 파일
- visualize_inference.ipynb : 결과 csv들의 데이터를 눈으로 확인할 수 있도록 plot 해주는 파일
- tuna.py : 만들어진 모델과 같은 환경에서 Optuna를 활용하여 hyperparameter를 최적화할 수 있도록 작성된 파일

## 💫 Final Model

---

- 의료 데이터의 high resolution 특성을 유지하기 위해 high resolution에 유리한 HRNet, DenseNet을 backbone(encoder)으로 두고, UNet++를 decoder로 사용함
- 각 모델에 의료 데이터 solution에서 자주 사용되는 augmentation들을 적용하고 추가로 seed를 바꾸면서 hard, soft 앙상블 진행
- 서버 GPU 용량 한계로 이미지를 1024 * 1024로 resize 후 학습

![Untitled (2)](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/9970d78b-1583-45fe-b296-14921e18ead4)


## 🔍 Reference 및 출처

---

- pytorch lightining : https://github.com/Lightning-AI/lightning
- hydra-zen : https://github.com/mit-ll-responsible-ai/hydra-zen
- mmsegmentation : https://github.com/open-mmlab/mmsegmentation
- segmentation_models.pytorch : https://github.com/qubvel/segmentation_models.pytorch
- HRNet : https://github.com/HRNet/HRNet-Semantic-Segmentation
- huggingface : https://huggingface.co/

## 📈 Result

---

<img width="1078" alt="Untitled (3)" src="https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/7c510893-9a75-48a9-93b0-87e11b78defb">



- Public Dice coefficient : 0.9725 / Private Dice coefficient : 0.9729
