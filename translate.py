from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from dataset import BilingualDataset, causal_mask
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import sys
import pandas as pd
from torchmetrics.text import BLEUScore

def get_all_sentences(ds, lang):
    for item in ds[lang]:
        yield item

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # train_ds_raw = pd.read_csv(config['datasource'])
    # val_ds_raw = pd.read_csv(config['val_datasource'])
    test_ds_raw = pd.read_csv(config['test_datasource'])

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, test_ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, test_ds_raw, config['lang_tgt'])
    #train_ds_raw = 
    # Keep 90% for training, 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in test_ds_raw[config['lang_src']]:
        src_ids = tokenizer_src.encode(item).ids
        tgt_ids = tokenizer_tgt.encode(item).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    return test_dataloader, tokenizer_src, tokenizer_tgt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)




def test(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, file_name):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    references = []
    metric =  BLEUScore()
    
    with torch.no_grad():
        with open(f'{file_name}.txt', 'w') as file:
            for batch in validation_ds:
                count += 1
                encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
    
                # check that the batch size is 1
                assert encoder_input.size(
                    0) == 1, "Batch size must be 1 for validation"
    
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
    
                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
                # source_texts.append(source_text)
                # expected.append(target_text)
                # predicted.append(model_out_text)
                # references.append([target_text])
                # Print the source, target and model output
    
                predicted = [model_out_text]
                references = [[target_text]]
                
                #print('-'*console_width)
                print(f"{f'SOURCE: ':>12}{source_text}")
                print(f"{f'TARGET: ':>12}{target_text}")
                print(f"{f'PREDICTED: ':>12}{model_out_text}")
                bleu = metric(predicted, references)

                file.write(f"{model_out_text}\t{bleu}\n")


# Define the device, tokenizers, and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model'], num_layers = config['num_layers'] , num_heads = config['num_heads']).to(device)

# Load the pretrained weights
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
test(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,config['end'])