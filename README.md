# Softex Predição fotovoltaica

## Instalação
```
pip install -r requirements.txt
```

## Base de dados

Baixar a base de dados em: https://drive.google.com/file/d/1qHBniPHa3IAojGU6il23Hh_Qd01ulgHp/view?usp=sharing

Extrair e armazenar em ./db

```
cd ./ai

python generate_dataset.py --layers_list 120 --input_labels "Radiacao_Avg" "Temp_Cel_Avg" "Potencia_FV_Avg" 
--output_labels "Potencia_FV_Avg" --input_steps 120 --output_steps 5 
```
## Treino
```
cd ./ai

python test.py --network 0 --layers_list 120 --input_labels "Radiacao_Avg" "Temp_Cel_Avg" "Potencia_FV_Avg" 
--output_labels "Potencia_FV_Avg" --input_steps 120 --output_steps 5 
```
> obs: --network 0 para MLP e --network 1 para LSTM.
## Teste

```
cd ./ai

python training.py --network 0 --layers_list 512 128 64 32 --input_labels "Radiacao_Avg" "Temp_Cel_Avg" "Potencia_FV_Avg" 
--output_labels "Potencia_FV_Avg" --input_steps 120 --output_steps 5
```
