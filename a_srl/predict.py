import logging
from typing import Iterable, Dict, List, Any

from allennlp.common.tqdm import Tqdm
from allennlp.common.params import Params
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator, DataIterator
from allennlp.models.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_model_predictions(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int) -> (Dict[str, Any], List):

    model.eval()
    model_predictions = []

    iterator = data_iterator(instances, num_epochs=1, cuda_device=cuda_device, for_training=False)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
    for batch in generator_tqdm:
        result = model(**batch)
        predictions = model.decode(result)
        model_predictions.extend(predictions["tags"])

    return model.get_metrics(), model_predictions