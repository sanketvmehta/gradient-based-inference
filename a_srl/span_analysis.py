import os
import errno
import sys
import argparse
import logging

from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.common import Params
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.data.instance import Instance
from allennlp.data import DatasetReader
from allennlp.models.archival import CONFIG_NAME

from a_srl.model.semantic_role_labeler import convert_bio_tags_to_conll_format
from a_srl.data.dataset_reader.dataset_utils.span_utils import *
from a_srl.utils.util import write_parse_trees_to_file
from a_srl.predict import *

logger = logging.getLogger(__name__)

def check_syntactic_consistency(instance: Instance, srl_tags: List[str], verb_index: Any,
                                spans_file, write_to_file: bool):

    spans = instance.fields["spans"].field_list #Constituent spans
    span_labels = instance.fields["span_labels"] #Constituent span labels

    metadata = instance.fields["metadata"].metadata

    if verb_index is None:
        return (-1, 0)

    srl_spans = srl_bio_tags_to_spans(tag_sequence=srl_tags)

    if metadata['span_labels'] is False:
        # Constituent parse data missing for this instance and we assume we have
        # exact match with syntactic constituents
        return (-2, len(srl_spans))
    else:
        span_labels = span_labels.labels

    syntactic_spans = get_syntactic_spans_from_SpanField(spans=spans, span_labels=span_labels)

    disagreeing_spans = set(srl_spans.keys()) - set(syntactic_spans.keys())

    if write_to_file:
        spans_file.write(str(syntactic_spans) + "\n")
        spans_file.write(str(srl_spans) + "\n")
        spans_file.write(str(disagreeing_spans) + "\n")
        spans_file.write("---------------------\n")
    return len(disagreeing_spans), len(srl_spans)

def analyze_gold_data(serialization_dir: str, instances: Iterable[Instance], split: str):

    spans_file_path = os.path.join(serialization_dir, "spans", "spans-" + split + ".txt")

    if not os.path.exists(os.path.dirname(spans_file_path)):
        try:
            os.makedirs(os.path.dirname(spans_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger.info("Writing syntactic constituents and srl spans to file: %s", spans_file_path)

    spans_file = open(spans_file_path, "w+")

    n_inst = len(instances)
    n_inst_with_parse = 0
    n_inst_with_predicate = 0
    n_inst_with_disagreement = 0

    n_args = 0
    n_args_with_missing_parse = 0
    n_args_with_disagreement = 0

    for instance in instances:

        fields = instance.fields
        try:
            verb_index = fields["verb_indicator"].labels.index(1)
        except ValueError:
            verb_index = None

        tags = fields["tags"].labels

        if verb_index is None:
            continue
        else:
            n_inst_with_predicate += 1

            result = check_syntactic_consistency(instance, srl_tags=tags, verb_index=verb_index,
                                                 spans_file=spans_file, write_to_file=True)

            res1 = result[0]
            res2 = result[1]

            if res1 == -2:
                # Parse information is missing
                n_args_with_missing_parse += res2
            elif res1 == 0:
                n_inst_with_parse += 1
                n_args += res2
            elif res1 > 0:
                n_inst_with_parse += 1
                n_inst_with_disagreement += 1
                n_args_with_disagreement += res1
                n_args += res2
            else:
                # res will be -1 which means verb predicate is missing
                logger.warning("Should never reach here!")
                pass

    spans_file.close()
    logger.info("-----------------------Instance Level Info (Gold)------------------------")
    logger.info("No. of instances = %d", n_inst)
    logger.info("No. of instances with verb predicates = %d", n_inst_with_predicate)
    logger.info("No. of instances with parse information = %d", n_inst_with_parse)
    logger.info("No. of instances with syntactic & srl spans disagreement = %d", n_inst_with_disagreement)
    logger.info("-----------------------Argument Level Info (Gold)------------------------")
    logger.info("No. of arguments = %d", n_args)
    logger.info("No. of arguments for instances with missing parse = %d", n_args_with_missing_parse)
    logger.info("No. of arguments with syntactic & srl spans disagreement = %d", n_args_with_disagreement)
    logger.info("-------------------------------------------------------------------------")

def analyze_model_predictions(serialization_dir: str, instances: Iterable[Instance],
                              model_predictions: List, split: str):

    spans_file_path = os.path.join(serialization_dir, "predicted-spans", "predicted-spans-" + split + ".txt")

    if not os.path.exists(os.path.dirname(spans_file_path)):
        try:
            os.makedirs(os.path.dirname(spans_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger.info("Writing model predicted srl spans (with gold syntactic constituents) to file: %s", spans_file_path)

    spans_file = open(spans_file_path, "w+")

    n_inst = len(instances)
    n_inst_exact = 0
    n_inst_with_missing_parse = 0
    n_inst_with_predicate = 0
    n_inst_with_parse = 0

    n_args = 0
    n_args_with_disagreement = 0
    n_args_with_missing_parse = 0

    n_args_gold = 0
    n_args_gold_with_missing_parse = 0
    n_args_gold_with_disagreement = 0

    n_props_with_only_verb = 0

    for instance, prediction in zip(instances, model_predictions):

        fields = instance.fields

        try:
            verb_index = fields["verb_indicator"].labels.index(1)
        except ValueError:
            verb_index = None

        if verb_index is None:
            continue
        else:
            n_inst_with_predicate += 1
            tags = fields["tags"].labels

            if tags.count("O") == len(tags) - 2:
                n_props_with_only_verb += 1

            result_prediction = check_syntactic_consistency(instance, prediction, verb_index, spans_file, True)
            res1 = result_prediction[0]
            res2 = result_prediction[1]

            result_actual = check_syntactic_consistency(instance, tags, verb_index, None, False)

            if res1 == -2:
                # Parse data is missing corresponding to this instance
                n_inst_with_missing_parse += 1
                n_args_with_missing_parse += res2
                n_args_gold_with_missing_parse += result_actual[1]
            elif res1 == -1:
                # There is no verb present in this instance
                logger.warning("Warning! Should never reach here!")
                n_inst_with_parse += 1
            elif res1 == 0:
                n_inst_exact += 1
                n_args += res2

                n_args_gold_with_disagreement += max(result_actual[0], 0)
                n_args_gold += result_actual[1]
                n_inst_with_parse += 1
            else:
                n_args += res2
                n_args_with_disagreement += res1

                n_args_gold += result_actual[1]
                n_args_gold_with_disagreement += max(result_actual[0], 0)
                n_inst_with_parse += 1

    spans_file.close()
    logger.info("-----------------------Instance Level Info------------------------------------")
    logger.info("No. of instances = %d", n_inst)
    logger.info("No. of instances with verb predicates = %d", n_inst_with_predicate)
    logger.info("No. of instances with parse information = %d", n_inst_with_parse)
    logger.info("No. of instances with missing parse = %d", n_inst_with_missing_parse)
    logger.info("-----------------------Argument/Instance Level Info (Predicted)---------------")
    logger.info("No. of arguments predicted = %d", n_args)
    logger.info("No. of arguments predicted corresponding to instances with missing parse data = %d", n_args_with_missing_parse)
    logger.info("No. of arguments in disagreement with syntactic spans = %d", n_args_with_disagreement)
    logger.info("No. of arguments in agreement with syntactic spans = %d", n_args - n_args_with_disagreement)
    logger.info("No. of instances with 100 perct. agreement with syntactic constituents = %d", n_inst_exact)
    logger.info("-----------------------Argument Level Info (Gold)-----------------------------")
    logger.info("No. of arguments in gold data (corresponding to instances with parse data) = %d", n_args_gold)
    logger.info("No. of arguments in gold data (corresponding to instances with missing parse data) = %d", n_args_gold_with_missing_parse)
    logger.info("No. of arguments in gold data with syntactic & srl spans disagreement  = %d", n_args_gold_with_disagreement)

    logger.info("No. of propositions with only verb predicate = %d", n_props_with_only_verb)
    logger.info("-------------------------------------------------------------------------")

def analyze_bio_violations(instances, model_predictions):

    # No. of violation at argument level
    n_b_i_violations = 0
    n_i_i_violations = 0
    n_o_i_violations = 0

    # No. of violations at instance level
    n_b_i_violations_at_inst = 0
    n_i_i_violations_at_inst = 0
    n_o_i_violations_at_inst = 0

    n_violations_inst = 0

    for instance, prediction in zip(instances, model_predictions):

        fields = instance.fields
        try:
            verb_index = fields["verb_indicator"].labels.index(1)
        except ValueError:
            verb_index = None

        if verb_index is None:
            continue

        spans, violations = bio_tags_to_spans(tag_sequence=prediction)

        if violations[0] > 0:
            n_b_i_violations += violations[0]
            n_b_i_violations_at_inst += 1
        if violations[1] > 0:
            n_i_i_violations += violations[1]
            n_i_i_violations_at_inst += 1
        if violations[2] > 0:
            n_o_i_violations += violations[2]
            n_o_i_violations_at_inst += 1

        if violations[0] > 0 or violations[1] > 0 or violations[2] > 0:
            n_violations_inst += 1

    logger.info("-------------------------Analyzing violations----------------------------")
    logger.info("No. of B-I violations = %d", n_b_i_violations)
    logger.info("No. of I-I violations = %d", n_i_i_violations)
    logger.info("No. of O-I violations = %d", n_o_i_violations)

    logger.info("No. of B-I violations (at instance level) = %d", n_b_i_violations_at_inst)
    logger.info("No. of I-I violations (at instance level) = %d", n_i_i_violations_at_inst)
    logger.info("No. of O-I violations (at instance level) = %d", n_o_i_violations_at_inst)

    logger.info("No. of instances with violations = %d", n_violations_inst)
    logger.info("-------------------------------------------------------------------------")

def verify_impl(model_predicitions):

    correct = 0

    for prediction in model_predicitions:

        conll_labels = convert_bio_tags_to_conll_format(prediction)
        conll_labels = ' '.join(conll_labels)

        conll_labels_new = bio_tags_to_conll_format(prediction)
        conll_labels_new = ' '.join(conll_labels_new)

        if conll_labels != conll_labels_new:
            print(conll_labels)
            print(conll_labels_new)
            print("-----------------------")
        else:
            correct += 1

    logger.info("Verifying BIO tags to CoNLL format conversion modules!")
    logger.info("No. of instances: %d", len(model_predicitions))
    logger.info("No. of correct: %d", correct)

def main(serialization_dir: str, cuda_device: int, split: str, overrides: str):

    """
    serialization_dir : str, required.
        The directory containing the serialized weights.
    cuda_device: int, default = -1
        The device to run the evaluation on.
    split: str, default = 'validation'
    """

    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)
    prepare_environment(config)

    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))

    data_path_key = split + '_data_path'
    data_path = config.pop(data_path_key)

    # Load the data
    logger.info("Reading "+split+" data from {}".format(data_path))

    instances = dataset_reader.read(data_path)

    write_parse_trees_to_file(serialization_dir=serialization_dir, instances=instances, split=split)

    analyze_gold_data(serialization_dir=serialization_dir, instances=instances,
                                split=split)

    model_predictions = get_model_predictions1(config=config, serialization_dir=serialization_dir,
                                              cuda_device=cuda_device, instances=instances)

    analyze_model_predictions(serialization_dir=serialization_dir, instances=instances,
                              model_predictions=model_predictions, split=split)

    analyze_bio_violations(instances, model_predictions)

    verify_impl(model_predicitions=model_predictions)
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write CONLL format SRL predictions"
                                                 " to file from a pretrained model.")
    parser.add_argument('--serialization_dir', type=str, help='The serialization directory.')
    parser.add_argument('--cuda_device', type=int, default=-1, help='The device to load the model onto.')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a HOCON structure used to override the experiment configuration.')
    parser.add_argument('--split', type=str, default="validation",
                        help='The split to evaluate the model. Used only is --evaluation_data_file is not provided.')
    parser.add_argument('--include_package', type=str, action='append', default=[],
                        help='additional packages to include')

    args = parser.parse_args()

    for package_name in args.include_package:
        import_submodules(package_name)

    main(args.serialization_dir, args.cuda_device, args.split, args.overrides)
