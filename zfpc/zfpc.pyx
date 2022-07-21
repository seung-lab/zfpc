"""
Optimally compressed partially corellated zfp streams container.

zfpc: zfp container

zfp doesn't optimally compress multi-channel data that
are not well correlated with each other. zfpc splits the
correlated data into different compressed streams and 
serializes the streams into a single file. You can then
treat the multiple compressed streams as a single compressed
file (including random access).

https://zfp.readthedocs.io/en/latest/faq.html#q-vfields
"""
import numpy as np
cimport numpy as np

from libcpp cimport bool as native_bool
from libcpp.vector cimport vector
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)

cdef extern from "zfpc.hpp" namespace "zfpc":
  cdef class ZfpcHeader:
    cdef str magic
    cdef uint8_t format_version
    cdef uint8_t data_type
    cdef uint16_t nx
    cdef uint16_t ny
    cdef uint16_t nz
    cdef uint16_t nw
    cdef uint8_t correlated_dims
    cdef bool c_order

    cdef ZfpcHeader fromchars(unsigned char* buf, size_t buflen)


def assemble_container(
  
):

