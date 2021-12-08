import logging
from launch import perform_launches

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    perform_launches()

