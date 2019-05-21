import os
import logging
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import Classifier
from finetune.datasets import Dataset, generic_download
from finetune.base_models.gpt.model import GPTModel
from finetune.base_models.gpt2.model import GPT2Model
from finetune.base_models.gpt.encoder import finetune_to_indico_attention_weights, to_spacy_attn
import matplotlib.pyplot as plt
import joblib as jl
logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"


class StanfordSentimentTreebank(Dataset):

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    def md5(self):
        return CHECKSUM

    def download(self):
        """
        Download Stanford Sentiment Treebank to data directory
        """
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        generic_download(
            url="https://s3.amazonaws.com/enso-data/SST-binary.csv",
            text_column="Text",
            target_column="Target",
            filename=SST_FILENAME
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=1000).dataframe
    model = Classifier(
        interpolate_pos_embed=False, 
        n_epochs=2,
        batch_size=2, 
        lr_warmup=0.1,
        val_size=0.0, 
        max_length=64,
        prefit_init=True,
        base_model=GPTModel,
        tensorboard_folder="./sst"
    )
    print(model.config.base_model_path)
    trainX, testX, trainY, testY = train_test_split(dataset.Text.values, dataset.Target.values, test_size=0.3, random_state=42)

    start_idx = 0
    num_samples = 100
    text = trainX[start_idx: start_idx + num_samples]
    text_labels = trainY[start_idx: start_idx + num_samples]

    model.fit(trainX, trainY)
    accuracy = np.mean(model.predict(text) == text_labels)
    print('Train Subset Accuracy: {:0.2f}'.format(accuracy))
    
    accuracy = np.mean(model.predict(trainX) == trainY)
    print('Train Accuracy: {:0.2f}'.format(accuracy))
    
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))

    attn_weights = model.attention_weights(text) # [batch, n_layer, n_heads, seq_len, seq_len]
    # one piece of text at a time
    for text_id, weights in enumerate(attn_weights):
        # print("weights", weights.shape)
        # each layer at a time
        token_weights = []
        for layer_weight in weights:
            # print("layer_weight", layer_weight.shape)
            attn = np.expand_dims(layer_weight, axis=0)
            # print("attn", attn.shape)
            output = finetune_to_indico_attention_weights([text[text_id]], attn, model.input_pipeline.text_encoder)[0]
            tokens = [text[text_id][s:e] for s, e in zip(output['token_starts'], output['token_ends'])]
            token_weights.append(output['attention_weights'])
        plt.imshow(np.array(token_weights), vmin=0, vmax=np.array(token_weights).max(), aspect='equal')
        plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=60)
        plt.tight_layout()
        plt.savefig('attn_2_epochs/attn{}.png'.format(text_id + start_idx))

