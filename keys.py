class ColorSchema:
    GRAYSCALE: str = 'grayscale'
    RGB: str = 'rgb'


class BenchmarkName:
    CHINESE: str = 'chinese'
    RUSSIAN: str = 'russian'
    BELGIUM: str = 'belgium'
    GERMANY:  str = 'germany'


class FileFolderPaths:
    CHINESE_TRAIN_ROOT: str = '../China-TSRD/TSRD-Train-Images/'
    CHINESE_TRAIN_ANNOTATIONS: str = '../China-TSRD/TSRD-Train-Annotation/TsignRecgTrain4170Annotation.txt'
    CHINESE_TEST_ROOT: str = '../China-TSRD/TSRD-Test-Images/'
    CHINESE_TEST_ANNOTATIONS: str = '../China-TSRD/TSRD-Test-Annotation/TsignRecgTest1994Annotation.txt'

    GERMAN_TRAIN_ROOT: str = '../german-GTRSD/'
    GERMAN_TRAIN_ANNOTATIONS: str = '../german-GTRSD/Train.csv'
    GERMAN_TEST_ROOT: str = '../german-GTRSD/'
    GERMAN_TEST_ANNOTATIONS: str = '../german-GTRSD/Test.csv'


class TableColumns:
    GERMAN_CLASS_COLUMN: str = 'ClassId'
    GERMAN_PATH_COLUMN: str = 'Path'


class Constants:
    MEAN_CHINESE_GRAYSCALE = 0.4255
    STD_CHINESE_GRAYSCALE = 0.2235
