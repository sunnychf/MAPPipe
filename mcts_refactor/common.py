import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DIFFPREP_DATASETS = {
    "obesity": 40966,
    "eeg": 1471,
    "ada_prior": 715,
    "pbcseq": 826,
    "jungle_chess_2pcs_raw_endgame_complete": 182,
    "wall-robot-navigation": 1504,
    "USCensus": 4534,
    "house_prices": 42165,
    "page-blocks": 30,
    "microaggregation2": 41156,
    "Run_or_walk_information": 41161,
    "connect-4": 40668,
    "shuttle": 40685,
    "mozilla4": 1045,
    "avila": 40701,
    "google": 41162,
    "pol": 722,
    "abalone": 183,
}


def get_dataset_name_by_id(dataset_id):
    for name, id_ in DIFFPREP_DATASETS.items():
        if id_ == dataset_id:
            return name
    return None
