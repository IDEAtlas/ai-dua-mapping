# -*- coding: utf-8 -*-
snap = None

def get_snappy():
    import esa_snappy
    return esa_snappy

def set_snappy(snappy_module):
    global snap
    snap = snappy_module
