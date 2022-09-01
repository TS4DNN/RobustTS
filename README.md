# Source Code of Towards Evaluating the Robustness of Test Selection Metrics for Deep Neural Networks



## Main Requirements
    - Tensorflow 2.3


## project structure
```
├── metrics                       # Test selection metrics
├── model_prepare                 # Train DNN models          
├── test_generation               # Methods for generating type1 and type2 data
├── utils                         # help functions
├── test_prioritization.py        # Perform fault detection
├── priori_retrain.py             # Perform repair
├── ts_run.py                     # Perform performance estimation
```

## Test data generation
```
python type1.py/type2.py -dataset mnist -model_type lenet1
```

## Fault detection, performance estimation, and model repair

```
python test_prioritization.py/ts_run.py/priori_retrain.py -dataset mnist -model_name lenet1 -save_path ... -data_name ori -attack_name ...
python ts_run.py -dataset mnist -model_name lenet1 -save_path ... -data_name ori --selected_layer 1
python riori_retrain.py -dataset mnist -model_name lenet1 -save_path ... -data_name ori -attack_name ...
```

## Others
- Traffic-Sign data: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
