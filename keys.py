class ColorSchema:
    GRAYSCALE: str = 'grayscale'
    RGB: str = 'rgb'


class BenchmarkName:
    CHINESE: str = 'chinese'
    RUSSIAN: str = 'russian'
    BELGIUM: str = 'belgium'
    GERMANY:  str = 'germany'


class NameKeys:
    BEST_CHECKPOINT: str = 'best'
    TRAINDIR: str = 'traindir'
    EXPLANATIONS: str = 'explanations_{}'
    VISUALIZATIONS: str = 'visualizations_{}'
    TRAIN_MODE_STRING: str = 'train'
    TEST_MODE_STRING: str = 'test'
    SOURCE_PNG: str = 'source.png'
    RESTORED_PNG: str = 'restored.png'


class FileFolderPaths:
    CHINESE_TRAIN_ROOT: str = '../China-TSRD/TSRD-Train-Images/'
    CHINESE_TRAIN_ANNOTATIONS: str = '../China-TSRD/TSRD-Train-Annotation/TsignRecgTrain4170Annotation.txt'
    CHINESE_TEST_ROOT: str = '../China-TSRD/TSRD-Test-Images/'
    CHINESE_TEST_ANNOTATIONS: str = '../China-TSRD/TSRD-Test-Annotation/TsignRecgTest1994Annotation.txt'

    GERMAN_TRAIN_ROOT: str = '../german-GTRSD/'
    GERMAN_TRAIN_ANNOTATIONS: str = '../german-GTRSD/Train.csv'
    GERMAN_TEST_ROOT: str = '../german-GTRSD/'
    GERMAN_TEST_ANNOTATIONS: str = '../german-GTRSD/Test.csv'

    BELGIUM_TRAIN_ROOT: str = '../belgium-TSC/BelgiumTSC_Training/Training'
    BELGIUM_TRAIN_ANNOTATIONS: str = None
    BELGIUM_TEST_ROOT: str = '../belgium-TSC/BelgiumTSC_Training/Training'
    BELGIUM_TEST_ANNOTATIONS: str = None

    RUSSIAN_TRAIN_ROOT: str = '../rtsd-r1/train/'
    RUSSIAN_TRAIN_ANNOTATIONS: str = '../rtsd-r1/gt_train.csv'
    RUSSIAN_TEST_ROOT: str = '../rtsd-r1/test/'
    RUSSIAN_TEST_ANNOTATIONS: str = '../rtsd-r1/gt_test.csv'


class TableColumns:
    GERMAN_CLASS_COLUMN: str = 'ClassId'
    GERMAN_PATH_COLUMN: str = 'Path'
    RUSSIAN_CLASS_COLUMN: str = 'class_number'
    RUSSIAN_PATH_COLUMN: str = 'filename'


class Constants:
    MEAN_CHINESE_GRAYSCALE = 0.4255
    STD_CHINESE_GRAYSCALE = 0.2235
