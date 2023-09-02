# 2023-Samsung-AI-Challenge-Camera-Invariant-Domain-Adaptation

왜곡이 없는(Rectilinear Source Domain) 이미지와 대응되는 레이블 정보를 활용하여, 
레이블이 존재하지 않는 왜곡된 영상(Fisheye* Target Domain)에서도 강인한 이미지 장면 분할(Semantic Segmentation) 인식을 수행하는 알고리즘 개발  
* Fisheye: 200도의 시야각(200° F.O.V)을 가지는 어안렌즈 카메라로 촬영된 이미지

# Data

├── <span style="color: #368CD6"> train </span>  
│   &nbsp; &nbsp; &nbsp;├── <span style="color: #1ED18B">train_source.csv</span>  
│   &nbsp; &nbsp; &nbsp;├── <span style="color: #368CD6">train_source_gt</span>  
│   &nbsp; &nbsp; &nbsp;├── <span style="color: #368CD6">train_source_image </span>  
│   &nbsp; &nbsp; &nbsp;├── <span style="color: #1ED18B">train_target.csv</span>  
│   &nbsp; &nbsp; &nbsp;└── <span style="color: #368CD6">train_target_image</span>      
├── <span style="color: #368CD6"> val </span>  
│   &nbsp; &nbsp; &nbsp;├── <span style="color: #1ED18B">val_source.csv</span>  
│   &nbsp; &nbsp; &nbsp;├── <span style="color: #368CD6">val_source_gt</span>  
│   &nbsp; &nbsp; &nbsp;└── <span style="color: #368CD6">val_source_image </span>    
└── <span style="color: #368CD6">test</span>  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;├── <span style="color: #368CD6">test_image </span>  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;└── <span style="color: #368CD6">test_image </span>  

## Sementic Segmentation Class

![Alt text](image.png)  

여기에 추가로 배경까지 총 13가지 Class를 가지고 있음.  
test시 효율성을 위해 RLE(Run-Length-Encoding)을 사용.  

ex) '1 3 10 5'는 픽셀 1,2,3 및 10,11,12,13,14가 마스크에 포함되어야 함을 의미