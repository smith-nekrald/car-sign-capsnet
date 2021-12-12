import logging
from launch import perform_launches


if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s %(funcName)s %(lineno)d : %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    perform_launches()

