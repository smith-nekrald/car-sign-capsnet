""" Specifies Keys/Names/Constants. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023


from typing import Optional
import os


class ColorSchema:
    """ Keys/Names for color schemas. 

    Attributes:
        GRAYSCALE: Key for grayscale color schema.
        RGB: Key for RGB color schema.
    """
    GRAYSCALE: str = 'grayscale'
    RGB: str = 'rgb'


class BenchmarkName:
    """ Keys/Names for benchmarks.

    Attributes:
        CHINESE: Key/Name for Chinese benchmark.
        RUSSIAN: Key/Name for Russian benchmark.
        BELGIUM: Key/Name for Belgian benchmark.
        GERMANY: Key/Name for German benchmark.
        ALL: Key/Name for sequentially training on all benchmarks.
        CHOICE_OPTIONS: The list with choices for CLI argument parsing.
    """
    CHINESE: str = 'chinese'
    RUSSIAN: str = 'russian'
    BELGIUM: str = 'belgium'
    GERMANY:  str = 'germany'

    ALL: str = 'all'
    CHOICE_OPTIONS: str = [CHINESE, RUSSIAN, BELGIUM, GERMANY, ALL]


class NameKeys:
    """ Naming-related keys and templates. 
    
    Attributes:
        BEST_CHECKPOINT: Name for the best checkpoint.
        TRAINDIR: Name for training folder.
        EXPLANATIONS: Name for the explanation folder.
        VISUALIZATIONS: Name for the visualizations folder. 
        TRAIN_MODE_STRING: Name/Key for the training mode.
        TEST_MODE_STRING: Name/Key for testing mode.
        SOURCE_PNG: Name pattern for source PNG file.
        RESTORED_PNG: Name pattern for restored PNG file.
        EXPLAIN_SRC_PNG: Name pattern for PNG file storing source image for LIME explaining.
        EXPLAIN_SUPERPIXELS_PNG: Name pattern for PNG file storing 
            LIME explanation (superpixels only).
        EXPLAIN_ALL_PNG:a Name pattern for PNG file storing LIME explanation (complete image).
        STATS_JSON: Name for the file with comparing statistics in JSON format.
        STATS_TEX: Name for the file with comparing statistics in LaTeX format.
        STATS_XLSX: Name for the file with comparing statistics in Excel format.
    """
    BEST_CHECKPOINT: str = 'best'
    TRAINDIR: str = 'traindir'
    EXPLANATIONS: str = 'explanations/{}'
    VISUALIZATIONS: str = 'visualizations/{}'
    TRAIN_MODE_STRING: str = 'train'
    TEST_MODE_STRING: str = 'test'
    SOURCE_PNG: str = 'source.png'
    RESTORED_PNG: str = 'restored.png'
    EXPLAIN_SRC_PNG: str = 'image-{}-explain-src.png'
    EXPLAIN_SUPERPIXELS_PNG: str = 'image-{}-explain-superpixels.png'
    EXPLAIN_ALL_PNG: str = 'image-{}-explain-all.png'
    STATS_JSON: str = 'stats.json'
    STATS_TEX: str = 'stats.tex'
    STATS_XLSX: str = 'stats.xlsx'


class FileFolderPaths:
    """ Names and paths for different file/folder locations.

    Attributes:
        CHINESE_TRAIN_ROOT: Path to root folder with train images for Chinese benchmark.
        CHINESE_TRAIN_ANNOTATIONS_ROOT: Path to root folder with train 
            annotations for Chinese benchmark.
        CHINESE_TRAIN_ANNOTATIONS: Path to file with train annotations for Chinese benchmark.
        CHINESE_TEST_ROOT: Path to root folder with test images for Chinese benchmark.
        CHINESE_TEST_ANNOTATIONS_ROOT: Path to root folder with test 
            annotations for Chinese benchmark.
        CHINESE_TEST_ANNOTATIONS: Path to file with test annotations for Chinese benchmark.
        GERMAN_TRAIN_ROOT: Path to root folder with train images for German benchmark.
        GERMAN_TRAIN_ANNOTATIONS: Path to file with train annotations for German benchmark.
        GERMAN_TEST_ROOT: Path to root folder with test images for German benchmark.
        GERMAN_TEST_ANNOTATIONS: Path to file with test annotations for German benchmark.
        BELGIUM_TRAIN_ROOT: Path to root folder with train images for Belgian benchmark.
        BELGIUM_TRAIN_ANNOTATIONS: Belgian annotations are included 
            in file names, therefore set to None.
        BELGIUM_TEST_ROOT: Path to root folder with test images for Belgian benchmark.
        BELGIUM_TEST_ANNOTATIONS: Belgian annotations are included
            in file names, therefore set to None.
        RUSSIAN_TRAIN_ROOT: Path to root folder with train images for Russian benchmark.
        RUSSIAN_TRAIN_ANNOTATIONS: Path to file with train annotations for Russian benchmark.
        RUSSIAN_TEST_ROOT: Path to root folder with test images for Russian benchmark.
        RUSSIAN_TEST_ANNOTATIONS: Path fo file with test annotations for Russian benchmark.
    """
    CHINESE_TRAIN_ROOT: str = '../benchmarks/China-TSRD/TSRD-Train-Images/'
    CHINESE_TRAIN_ANNOTATIONS_ROOT: str = '../benchmarks/China-TSRD/TSRD-Train-Annotation'
    CHINESE_TRAIN_ANNOTATIONS: str = os.path.join(CHINESE_TRAIN_ANNOTATIONS_ROOT,  
                                                  'TsignRecgTrain4170Annotation.txt')
    CHINESE_TEST_ROOT: str = '../benchmarks/China-TSRD/TSRD-Test-Images/'
    CHINESE_TEST_ANNOTATIONS_ROOT: str = '../benchmarks/China-TSRD/TSRD-Test-Annotation'
    CHINESE_TEST_ANNOTATIONS: str = os.path.join(CHINESE_TEST_ANNOTATIONS_ROOT, 
                                                 'TsignRecgTest1994Annotation.txt')

    GERMAN_TRAIN_ROOT: str = '../benchmarks/german-GTRSD/'
    GERMAN_TRAIN_ANNOTATIONS: str = '../benchmarks/german-GTRSD/Train.csv'
    GERMAN_TEST_ROOT: str = '../benchmarks/german-GTRSD/'
    GERMAN_TEST_ANNOTATIONS: str = '../benchmarks/german-GTRSD/Test.csv'

    BELGIUM_TRAIN_ROOT: str = '../benchmarks/belgium-TSC/BelgiumTSC_Training/Training'
    BELGIUM_TRAIN_ANNOTATIONS: Optional[str] = None
    BELGIUM_TEST_ROOT: str = '../benchmarks/belgium-TSC/BelgiumTSC_Training/Training'
    BELGIUM_TEST_ANNOTATIONS: Optional[str] = None

    RUSSIAN_TRAIN_ROOT: str = '../benchmarks/rtsd-r1/train/'
    RUSSIAN_TRAIN_ANNOTATIONS: str = '../benchmarks/rtsd-r1/gt_train.csv'
    RUSSIAN_TEST_ROOT: str = '../benchmarks/rtsd-r1/test/'
    RUSSIAN_TEST_ANNOTATIONS: str = '../benchmarks/rtsd-r1/gt_test.csv'


class TableColumns:
    """ Column names in tables. 

    Attributes:
        GERMAN_CLASS_COLUMN: The name of the column containing classes in German benchmark.
        GERMAN_PATH_COLUMN: The name of the column containing paths in German benchmark.
        RUSSIAN_CLASS_COLUMN: The name of the column containing classes in Russian benchmark.
        RUSSIAN_PATH_COLUMN: The name of the column containing paths in Russian benchmark.
    """
    GERMAN_CLASS_COLUMN: str = 'ClassId'
    GERMAN_PATH_COLUMN: str = 'Path'
    RUSSIAN_CLASS_COLUMN: str = 'class_number'
    RUSSIAN_PATH_COLUMN: str = 'filename'


class StatsTableKeys:
    """ Column names in summary tables.

    Attributes:
        DATASET: Names the column with benchmark name.
        EPOCH_ID: Names the column with id of optimal epoch.
        TEST_ACCURACY: Names the column with test accuracies.
        TRAIN_ACCURACY: Names the column with train accuracies.
        TEST_LOSS: Names the column with test losses.
        TRAIN_LOSS: Names the column with train losses.
    """
    DATASET: str = 'Benchmark'
    EPOCH_ID: str = 'Epoch Id'
    TEST_ACCURACY: str = 'Test Accuracy'
    TRAIN_ACCURACY: str = 'Train Accuracy'
    TEST_LOSS: str = 'Test Loss'
    TRAIN_LOSS: str = 'Train Loss'


class Constants:
    """ Some helpful pre-computed constants. 

    Attributes:
        MEAN_CHINESE_GRAYSCALE: Mean value used for normalizing 
            chinese benchmark in grayscale format.
        STD_CHINESE_GRAYSCALE: Std value used for normalizing 
            chinese benchmark in grayscale format.
    """
    MEAN_CHINESE_GRAYSCALE: float = 0.4255
    STD_CHINESE_GRAYSCALE: float = 0.2235

