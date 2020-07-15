import os
import errno
import argparse
import sys
import logging
import torch
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

from typing import Dict, Any
from allennlp.models.archival import load_archive
from allennlp.models.archival import CONFIG_NAME, _WEIGHTS_NAME
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment, import_submodules, prepare_global_logging
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, get_text_field_mask

from a_srl.predict import get_model_predictions
from a_srl.span_analysis import analyze_model_predictions, analyze_gold_data, analyze_bio_violations

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Disable some of the more verbose logging statements
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

def eval(f_gold, f_predicted, f_script_path = 'evaluation/'):

    eval_cmd = ['bash', f_script_path + 'eval.sh', f_gold, f_predicted]

    print('eval_cmd: ' + ' '.join(eval_cmd))

    try:
        eval_out = subprocess.check_output(eval_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        eval_out = None

    # print(eval_out)

    if eval_out:
        return extract_eval_out(eval_out=eval_out)
    else:
        return 0.0, 0.0, 0.0

def extract_eval_out(eval_out):

    eval_out = eval_out.decode("utf-8")
    eval_out = eval_out.split('\n')

    results = eval_out[8].split(' ')
    results = [x for x in results if x != '']

    if len(results) == 7:
        f1_score = float(results[-1])
        recall = float(results[-2])
        prec = float(results[-3])
    else:
        f1_score = 0.0
        recall = 0.0
        prec = 0.0

    # logger.info("F1_score: %d", f1_score)
    # logger.info("Precision: %d", prec)
    # logger.info("Recall: %d", recall)
    return f1_score, prec, recall

def write_predictions_viterbi_decoded(serialization_dir, split, epoch, predicted_tags,
                              vocab: Vocabulary,
                              tokens: Dict[str, torch.LongTensor],
                                verb_indicator: torch.LongTensor,
                                tags: torch.LongTensor = None,
                                pos_tags: torch.LongTensor = None,
                                spans: torch.LongTensor = None,
                                span_labels: torch.LongTensor = None,
                                metadata: Any = None):

    prediction_file_path = os.path.join(serialization_dir, "predictions", "predictions-" + split + "-" + str(epoch) + ".txt")
    gold_file_path = os.path.join(serialization_dir, "predictions", "gold-" + split + "-" + str(epoch) + ".txt")

    if not os.path.exists(os.path.dirname(prediction_file_path)):
        try:
            os.makedirs(os.path.dirname(prediction_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # logger.info("Writing gold srl tags (in conll file format) to %s", gold_file_path)
    # logger.info("Writing predicted srl tags (in conll file format) to %s", prediction_file_path)

    prediction_file = open(prediction_file_path, "a+")
    gold_file = open(gold_file_path, "a+")

    sentences = tokens["tokens"]
    mask = get_text_field_mask(tokens)
    sentence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()

    for sentence, _gold_tags, _verb_indicator, _length, _predicted_tags in zip(sentences.data.cpu(), tags.data.cpu(),
                                                            verb_indicator.data.cpu(), sentence_lengths, predicted_tags.data.cpu()):
        tokens = [vocab.get_token_from_index(x, namespace="tokens").__str__()
                  for x in sentence[:_length]]
        gold_labels = [vocab.get_token_from_index(x, namespace="labels")
                         for x in _gold_tags[:_length]]
        _verb_indicator = [x for x in _verb_indicator[: _length]]

        prediction = [vocab.get_token_from_index(x, namespace="labels")
                         for x in _predicted_tags[:_length]]

        try:
            verb_index = _verb_indicator.index(1)
        except ValueError:
            verb_index = None

        # Defined in semantic_role_labeler model implementation
        write_to_conll_eval_file(prediction_file=prediction_file, gold_file=gold_file,
                                 verb_index=verb_index, sentence=tokens, prediction=prediction,
                                 gold_labels=gold_labels)
    prediction_file.close()
    gold_file.close()

def write_predictions_viterbi(serialization_dir, split, epoch, model_predictions,
                              vocab: Vocabulary,
                              tokens: Dict[str, torch.LongTensor],
                                verb_indicator: torch.LongTensor,
                                tags: torch.LongTensor = None,
                                pos_tags: torch.LongTensor = None,
                                spans: torch.LongTensor = None,
                                span_labels: torch.LongTensor = None,
                                metadata: Any = None):


    prediction_file_path = os.path.join(serialization_dir, "predictions", "predictions-" + split + "-" + str(epoch) + ".txt")
    gold_file_path = os.path.join(serialization_dir, "predictions", "gold-" + split + "-" + str(epoch) + ".txt")

    if not os.path.exists(os.path.dirname(prediction_file_path)):
        try:
            os.makedirs(os.path.dirname(prediction_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # logger.info("Writing gold srl tags (in conll file format) to %s", gold_file_path)
    # logger.info("Writing predicted srl tags (in conll file format) to %s", prediction_file_path)

    prediction_file = open(prediction_file_path, "a+")
    gold_file = open(gold_file_path, "a+")

    sentences = tokens["tokens"]
    mask = get_text_field_mask(tokens)
    sentence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()

    for sentence, _gold_tags, _verb_indicator, _length, prediction in zip(sentences.data.cpu(), tags.data.cpu(),
                                                            verb_indicator.data.cpu(), sentence_lengths, model_predictions):
        tokens = [vocab.get_token_from_index(x, namespace="tokens").__str__()
                  for x in sentence[:_length]]
        # logger.info("Tokens: %s", ' '.join(tokens))
        gold_labels = [vocab.get_token_from_index(x, namespace="labels")
                         for x in _gold_tags[:_length]]
        _verb_indicator = [x for x in _verb_indicator[: _length]]

        try:
            verb_index = _verb_indicator.index(1)
        except ValueError:
            verb_index = None

        # Defined in semantic_role_labeler model implementation
        write_to_conll_eval_file(prediction_file=prediction_file, gold_file=gold_file,
                                 verb_index=verb_index, sentence=tokens, prediction=prediction,
                                 gold_labels=gold_labels)
    prediction_file.close()
    gold_file.close()

def write_predictions(serialization_dir, instances, model_predictions, split, epoch = None):

    if epoch:
        prediction_file_path = os.path.join(serialization_dir, "predictions", "predictions-" + split + "-" + str(epoch) + ".txt")
        gold_file_path = os.path.join(serialization_dir, "predictions", "gold-" + split + "-" + str(epoch) + ".txt")
    else:
        prediction_file_path = os.path.join(serialization_dir, "predictions", "predictions-"+split+".txt")
        gold_file_path = os.path.join(serialization_dir, "predictions", "gold-"+split+".txt")

    if not os.path.exists(os.path.dirname(prediction_file_path)):
        try:
            os.makedirs(os.path.dirname(prediction_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger.info("Writing gold srl tags (in conll file format) to %s", gold_file_path)
    logger.info("Writing predicted srl tags (in conll file format) to %s", prediction_file_path)

    prediction_file = open(prediction_file_path, "a+")
    gold_file = open(gold_file_path, "a+")

    for instance, prediction in zip(instances, model_predictions):
        fields = instance.fields
        try:
            # Most sentences have a verbal predicate, but not all.
            verb_index = fields["verb_indicator"].labels.index(1)
        except ValueError:
            verb_index = None

        gold_labels = fields["tags"].labels
        sentence = fields["tokens"].tokens

        # Defined in semantic_role_labeler model implementation
        write_to_conll_eval_file(prediction_file, gold_file,
                                 verb_index, sentence, prediction, gold_labels)
    prediction_file.close()
    gold_file.close()

def main(serialization_dir, evaluation_data_file, split, cuda_device, weights_file, overrides):

    archive_file = os.path.join(serialization_dir, "model.tar.gz")

    logging_dir = os.path.join(serialization_dir, "logging")

    if os.path.isfile(archive_file):
        weights_file = None
        archive = load_archive(archive_file, cuda_device, overrides, weights_file)
        config = archive.config
        prepare_environment(config)
        prepare_global_logging(logging_dir, file_friendly_logging=False, file_name=split)
        model = archive.model
    else:
        # Load config
        config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)
        prepare_environment(config)
        prepare_global_logging(logging_dir, file_friendly_logging=False, file_name=split)

        if weights_file:
            weights_path = os.path.join(serialization_dir, weights_file)
        else:
            weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)

        logger.info("Using weights_file located at : %s", weights_path)
        # Instantiate model. Use a duplicate of the config, as it will get consumed.
        model = Model.load(config.duplicate(),
                           weights_file=weights_path,
                           serialization_dir=serialization_dir,
                           cuda_device=cuda_device)

    # Eval mode ON
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))

    if evaluation_data_file is None:
        logger.info("--evaluation_data_file not provided. So using --split=%s to read data", split)
        data_path_key = split + '_data_path'
        evaluation_data_path = config.pop(data_path_key)
    else:
        evaluation_data_path = evaluation_data_file

    logger.info("Reading evaluation data from %s", evaluation_data_path)

    instances = dataset_reader.read(evaluation_data_path)
    logger.info("No. of instances = %d", len(instances))

    iterator = BasicIterator(batch_size=128)
    iterator.index_with(model.vocab)

    metrics, model_predictions = get_model_predictions(model, instances, iterator, args.cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    write_predictions(serialization_dir=serialization_dir, instances=instances,
                 model_predictions=model_predictions, split=split)

    analyze_gold_data(serialization_dir=serialization_dir, instances=instances, split=split)

    analyze_model_predictions(serialization_dir=serialization_dir, instances=instances,
                              model_predictions=model_predictions, split=split)

    analyze_bio_violations(instances=instances, model_predictions=model_predictions)

if __name__ == "__main__":

    description = "Evaluate the specified model + dataset and write " \
                  "CONLL format SRL predictions to file."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--serialization_dir', type=str, help='The path to a serialized directory which contains an archived trained model (if any).')
    parser.add_argument('--evaluation_data_file', type=str, default=None, help='The path to the file containing the evaluation data.')
    parser.add_argument('--split', type=str, default="validation", help='The split to evaluate the model. Used only is --evaluation_data_file is not provided.')
    parser.add_argument('--cuda_device', type=int, default=0, help='The id of GPU to use (if any)')
    parser.add_argument('--weights_file', type=str, default="best.th", help='A path that overrides which weights file to use.')
    parser.add_argument('-o', '--overrides', type=str, default="", help='a HOCON structure used to override the experiment configuration.')
    parser.add_argument('--include_package', type=str, action='append', default=[], help='additional packages to include')

    args = parser.parse_args()

    for package_name in args.include_package:
        import_submodules(package_name)

    main(args.serialization_dir, args.evaluation_data_file, args.split, args.cuda_device, args.weights_file, args.overrides)