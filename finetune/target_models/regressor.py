import tensorflow as tf
import numpy as np

from finetune.base import BaseModel
from finetune.encoding.target_encoders import RegressionEncoder
from finetune.nn.target_blocks import regressor
from finetune.input_pipeline import BasePipeline


class RegressionPipeline(BasePipeline):
    def _target_encoder(self):
        return RegressionEncoder()


class Regressor(BaseModel):
    """ 
    Regresses one or more floating point values given a single document.

    For a full list of configuration options, see `finetune.config`.

    :param config: A config object generated by `finetune.config.get_config` or None (for default config).
    :param \**kwargs: key-value pairs of config items to override.
    """

    def _get_input_pipeline(self):
        return RegressionPipeline(self.config)

    def featurize(self, X):
        """
        Embeds inputs in learned feature space. Can be called before or after calling :meth:`finetune`.

        :param X: list or array of text to embed.
        :returns: np.array of features of shape (n_examples, embedding_size).
        """
        return self._featurize(X)

    def predict(self, X):
        """
        Produces a list of most likely class labels as determined by the fine-tuned model.

        :param X: list or array of text to embed.
        :returns: list of class labels.
        """
        all_labels=[]
        for _, start_of_doc, end_of_doc, label, _ in self.process_long_sequence(X):
            if start_of_doc:
                # if this is the first chunk in a document, start accumulating from scratch
                doc_labels = []
                
            doc_labels.append(label)

            if end_of_doc:
                # last chunk in a document
                means = np.mean(doc_labels, axis=0)
                label = self.input_pipeline.label_encoder.inverse_transform([means])
                all_labels.append(list(label))
        return all_labels

    def predict_proba(self, X):
        """
        Produces a probability distribution over classes for each example in X.

        :param X: list or array of text to embed.
        :returns: list of dictionaries.  Each dictionary maps from a class label to its assigned class probability.
        """
        raise AttributeError("`Regressor` model does not support `predict_proba`.")

    def finetune(self, X, Y=None, batch_size=None):
        """
        :param X: list or array of text.
        :param Y: floating point targets
        :param batch_size: integer number of examples per batch. When N_GPUS > 1, this number
                           corresponds to the number of training examples provided to each GPU.
        """
        return super().finetune(X, Y=Y, batch_size=batch_size)

    @staticmethod
    def _target_model(config, featurizer_state, targets, n_outputs, train=False, reuse=None, **kwargs):
        return regressor(
            hidden=featurizer_state['features'],
            targets=targets, 
            n_targets=n_outputs,
            config=config,
            train=train, 
            reuse=reuse, 
            **kwargs
        )

    def _predict_op(self, logits, **kwargs):
        return logits

    def _predict_proba_op(self, logits, **kwargs):
        return logits
