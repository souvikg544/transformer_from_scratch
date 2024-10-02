Transformers from Scratch in Pytorch

The work has been done using the ted talks corpus. If using any other corpus, one can set up the respective csv files from palyground.ipynb. Also we demonstrate several tests with bleu score on the playground.

Usage - 
1. Make the suitable training configuration and train the transformer -
   ```
   python train.py
   ```

2. Test
   ```
   python translate.py
   ```

The files trans1.txt,trans2.txt and trans3.txt are the test scores for the three experiments. Sorry for not naming them testbleu.txt .

Experiments - 

- trans1 :
  ```
  "d_model": 512,
  "num_layers" : 6,
  "num_heads" : 8,
  ```

- trans2 :
  ```
    "d_model": 768,
    "num_layers" : 8,
    "num_heads" : 8,
    ```

- trans3 :
  ```
    "d_model": 768,
    "num_layers" : 8,
    "num_heads" : 12,
```