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


