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

## Reference
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/lucidrains/linformer

