# Long Sequence Time Forecasting(LSTF)

## Project explanation
This project is to forecast long sequence time data on Stock Price. We also forecast Electricity Power Consumption to compare deep learning models. We built Informer for the main model and added Rinformer, Seq2Seq and CNN. We found that Informer is powerful to forecast long sequences compare to other models.

## Framework
1) Preprocessing data for Stock Price and Electricity Power Consumption data.
2) Spliting data into Train and Test set.
3) Building Data Loaders.
4) Building Models(Informer, Rinformer, Seq2Seq and CNN).
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


## Reference
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/lucidrains/linformer

