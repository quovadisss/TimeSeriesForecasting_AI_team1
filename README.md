# Long Sequence Time Forecasting(LSTF)

## Project explanation
This project is to forecast long sequence time data on Stock Price. We also forecast Electricity Power Consumption to compare deep learning models. We built Informer for the main model and added Linformer, Seq2Seq and CNN. We found that Informer is powerful to forecast long sequences compare to other models.

## Framework
1) Preprocessing data for Stock Price and Electricity Power Consumption data.
2) Spliting data into Train and Test set.
3) Building Data Loaders.
4) Building Models(Informer, Linformer, Seq2Seq and CNN).
5) Inference.
6) Plotting all the results and finding insights.

## Usage
1) ```python main.py --model informer --gpu_id 0 --output_attention```
2) or you can just execute run.sh file.

## Plots(Example)
- AMD Price in NASDAQ 
    - Forecasting 1 hour / 7 hours / 14 hours / 30 hours
    - CNN
![AMD_CNN_type0](https://user-images.githubusercontent.com/56912449/121091205-9e96e880-c824-11eb-84ba-07106fe6433d.png)
![AMD_CNN_type1](https://user-images.githubusercontent.com/56912449/121091629-444a5780-c825-11eb-8f5f-7bd011a04982.png)
![AMD_CNN_type2](https://user-images.githubusercontent.com/56912449/121091661-4dd3bf80-c825-11eb-8f96-30eca7697073.png)
![AMD_CNN_type3](https://user-images.githubusercontent.com/56912449/121091667-50361980-c825-11eb-946f-739554a13c52.png)
    - S2S
![AMD_S2S_type0](https://user-images.githubusercontent.com/56912449/121091908-b1f68380-c825-11eb-8537-3b936e3529a4.png)
![AMD_S2S_type1](https://user-images.githubusercontent.com/56912449/121091916-b4f17400-c825-11eb-89e3-3219defe1141.png)
![AMD_S2S_type2](https://user-images.githubusercontent.com/56912449/121091923-b7ec6480-c825-11eb-92b6-9ad3d17f9080.png)
![AMD_S2S_type3](https://user-images.githubusercontent.com/56912449/121091930-b9b62800-c825-11eb-953e-a61da607f8a5.png)
    - Informer
![AMD_informer_type0](https://user-images.githubusercontent.com/56912449/121091986-d7838d00-c825-11eb-8b48-206ff76fc8c2.png)
![AMD_informer_type1](https://user-images.githubusercontent.com/56912449/121091989-d94d5080-c825-11eb-8ff1-31998eafd981.png)
![AMD_informer_type2](https://user-images.githubusercontent.com/56912449/121091993-da7e7d80-c825-11eb-97f6-19a2bb3c56ae.png)
![AMD_informer_type3](https://user-images.githubusercontent.com/56912449/121091995-da7e7d80-c825-11eb-96aa-947945a91845.png)

## Metric
<img width="1170" alt="스크린샷 2021-06-08 오전 6 55 19" src="https://user-images.githubusercontent.com/56912449/121092458-87f19100-c826-11eb-8894-f3e8c486fbc4.png">

## Reference
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/lucidrains/linformer

