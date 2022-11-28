# Edge - PV Power Forecast

## Installation
```
pip install -r requirements.txt
```

## Dataset
Download the database at: https://drive.google.com/file/d/1o_h3Kc-FdNPv11Tg10VC4CSNixCFDWPM/view?usp=share_link

Extract and store in ./db

```
cd ./ai

python generate_dataset.py --layers_list [120] --input_labels "Radiacao_Avg" "Temp_Cel_Avg" "Potencia_FV_Avg" 
--output_labels "Potencia_FV_Avg" --input_steps 120 --output_steps 5 
```

## Models
Download the models and normalizators at: https://drive.google.com/file/d/18GBEGzRjh_b6ASN_o4r5ZiLYbm9oR8ZR/view?usp=sharing

Extract and store in ./db

## Training
```
cd ./ai

python training.py --network 0 --layers_list [120] --input_labels "Radiacao_Avg" "Temp_Cel_Avg" "Potencia_FV_Avg" 
--output_labels "Potencia_FV_Avg" --input_steps 120 --output_steps 5 
```
> obs: --network 0 for MLP, --network 1 for RNN, and --network 2 for LSTM.
## Test

```
cd ./ai

python test.py --network 0 --layers_list [120] --input_labels "Radiacao_Avg" "Temp_Cel_Avg" "Potencia_FV_Avg" 
--output_labels "Potencia_FV_Avg" --input_steps 120 --output_steps 5
```

