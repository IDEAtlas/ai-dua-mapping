import nicesnappy.utils as utils
from nicesnappy.operators import *

def initialize():
    snappy = utils.get_snappy()
    utils.set_snappy(snappy)

def initialize_with_module(snappy):
    utils.set_snappy(snappy)
