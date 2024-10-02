Transformers from Scratch in Pytorch

The work has been done using the ted talks corpus. If using any other corpus, one can set up the respective csv files from palyground.ipynb. Also we demonstrate several tests with bleu score on the playground.

Usage - 
1. Make the suitable training configuration and train the transformer -
   ```
   python train.py
   ```

2. Test - Use the following [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/souvik_ghosh_students_iiit_ac_in/ESr0VzFJ9-9Hta4U27FcFl4Bj0nLfeoWyAhOHCsosCIVFQ?e=6KG4s3) to download the trans1 model weight. All other model weights are available too in the corresponding folder.
3. 
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
