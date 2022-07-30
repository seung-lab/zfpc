from typing import Tuple, List
import enum
import numpy as np

import zfpy

__all__ = [ "compress", "decompress" ]

DATA_TYPES = {
  None: 0,
  np.int32: 1,
  np.int64: 2,
  np.float32: 3,
  np.float64: 4,
}
for k,v in list(DATA_TYPES.items()):
  DATA_TYPES[v] = k
  DATA_TYPES[np.dtype(k)] = v

CorrelatedDimsType = Tuple[bool,bool,bool,bool]

class ZfpModes(enum.IntEnum):
  NULL = 0
  EXPERT = 1
  FIXED_RATE = 2
  FIXED_PRECISION = 3
  FIXED_ACCURACY = 4
  REVERSIBLE = 5

def which_mode(tolerance:float, rate:float, precision:float) -> int:
  """Corresponds to subset of enum in zfp.h"""
  if sum([ tolerance >= 0, rate >= 0, precision >= 0 ]) > 1:
    raise ValueError(
      "Only one of tolerance, rate, and precision can be specified (i.e. > -1) at once."
      "tolerance: {tolerance}, rate: {rate}, precision: {precision}"
    )

  if tolerance > -1:
    return ZfpModes.FIXED_ACCURACY
  elif precision > -1:
    return ZfpModes.FIXED_PRECISION
  elif rate > -1:
    return ZfpModes.FIXED_RATE
  else:
    return ZfpModes.REVERSIBLE

class DecodingError(Exception):
  pass

class ZfpcHeader:
  header_size:int = 23
  format_version:int = 0
  magic:str = b'zfpc'

  def __init__(
    self, 
    nx:int, ny:int, nz:int, nw:int,
    data_type:np.dtype, 
    c_order:bool, mode:int,
    correlated_dims:CorrelatedDimsType
  ):
    self.nx = int(nx)
    self.ny = int(ny)
    self.nz = int(nz)
    self.nw = int(nw)
    self.data_type = data_type
    self.c_order = bool(c_order)
    self.mode = int(mode)
    self.correlated_dims = correlated_dims

  @property
  def random_access(self) -> bool:
    return self.mode == ZfpModes.FIXED_RATE

  def shape(self):
    shape = [self.nx, self.ny, self.nz, self.nw]
    while len(shape) and shape[-1] == 0:
      shape.pop()

    return shape

  def tobytes(self) -> bytes:
    shape = np.array([self.nx, self.ny, self.nz, self.nw], dtype=np.uint32)
    data_type = (DATA_TYPES[self.data_type] & 0b111)
    data_type |= ((int(self.mode) & 0b111) << 3)
    data_type |= (int(bool(self.c_order)) << 7)

    correlated_dims = 0
    for i, is_correlated in enumerate(self.correlated_dims):
      correlated_dims |= (is_correlated << i)

    return (
        self.magic
      + int.to_bytes(self.format_version, length=1, byteorder="little", signed=False)
      + int.to_bytes(data_type, length=1, byteorder="little", signed=False)
      + shape.tobytes()
      + int.to_bytes(correlated_dims, length=1, byteorder="little", signed=False)
    )

  @classmethod
  def frombytes(self, binary:bytes):
    if len(binary) < ZfpcHeader.header_size:
      raise DecodingError(f"Buffer too small to decode header.")

    magic = binary[:len(ZfpcHeader.magic)]
    if magic != ZfpcHeader.magic:
      raise DecodingError(f"Magic number did not match. Got: {magic}")

    format_version = int(binary[4])
    if format_version > ZfpcHeader.format_version:
      raise DecodingError(f"Only format versions {ZfpcHeader.format_version} and lower are supported. Got: {format_version}")

    data_type = int(binary[5])
    c_order = bool(data_type >> 7)
    mode = (data_type >> 3) & 0b111
    data_type = data_type & 0b00000111

    nx, ny, nz, nw = list(np.frombuffer(binary[6:22], dtype=np.uint32))
    correlated_dims = [ bool((binary[22] >> i) & 0b1) for i in range(4) ]

    return ZfpcHeader(
      nx, ny, nz, nw, 
      DATA_TYPES[data_type], 
      c_order, mode,
      correlated_dims
    )

  def num_streams(self):
    shape = [ self.nx, self.ny, self.nz, self.nw ]
    num_streams = 1
    # size 0 is treated as the dimension does not exist. Zeros should
    # only occur on the rhs.
    for i in range(4):
      if (shape[i] > 1 and (self.correlated_dims[i] == False)):
        num_streams *= shape[i]
    return num_streams

def compute_slices(header):
  shape = [ header.nx, header.ny, header.nz, header.nw ]
  ndim = sum([ si > 0 for si in shape ])
  shape = [ si if si > 0 else 1 for si in shape ]

  x_t = lambda x: slice(None) if header.correlated_dims[0] else x
  y_t = lambda y: slice(None) if header.correlated_dims[1] else y
  z_t = lambda z: slice(None) if header.correlated_dims[2] else z
  w_t = lambda w: slice(None) if header.correlated_dims[3] else w
  slice_template = lambda x,y,z,w: ( x_t(x), y_t(y), z_t(z), w_t(w) )

  dim_iter = lambda i: [0] if header.correlated_dims[i] else range(shape[i])

  for w in dim_iter(3):
    for z in dim_iter(2):
      for y in dim_iter(1):
        for x in dim_iter(0):
          yield tuple(slice_template(x,y,z,w)[:ndim])

def compress(
  data:np.ndarray, 
  correlated_dims:CorrelatedDimsType = [True, True, True, True],
  tolerance:float = -1.0,
  precision:float = -1.0,
  rate:float = -1.0,
) -> bytes:
  """
  Compress a numpy array into a zfpc stream.

  A zfpc stream consists of multiple zfp streams.
  The data are split apart in order to optimally 
  compress only the correlated dimensions of the 
  data. For example, a 3D vector field might
  consist of 2d planes of indpendent X and Y 
  vectors which are optimally compressed as 
  independant planes of each channel.

  The number of internal zfp streams is equal 
  to the product of the uncorrelated dimensions. 
  For example:
  [T,T,T,T] -> 1 stream
  [T,T,T,F] -> # of channels streams
  [T,T,F,F] -> # of channels streams x nz

  data: A 1-4D numpy array in either C or F order.
  correlated_dims: A 4D boolean array with True values
    indicating which dimensions are corellated. 

    In the 3D vector field case, this would be
    [True,True,False,False] as each 2D plane is
    smoothly varying but each slice and each channel
    are independent.

  You can only specify one of:

  tolerance: absolute tolerable error from lossy compression
    (aka fixed accuracy mode)
  rate: number of bits per field, enables random access
  precision: number of uncompressed bits for transform 
    coefficients, affects relative error.

  If none of these coefficients are set, the stream is set
  to reversible (lossless compression) mode.

  You can read more about these options here:
  https://zfp.readthedocs.io/en/latest/modes.html#fixed-rate-mode
  """
  if correlated_dims == [False]*4:
    raise ValueError(
      "All uncorrlated dims result in an independent stream for each voxel. "
      "At least one dimension must be correlated."
    )

  shape = [0,0,0,0]
  for i, sz in enumerate(data.shape):
    shape[i] = sz

  header = ZfpcHeader(
    shape[0], shape[1], shape[2], shape[3],
    data_type=data.dtype, 
    c_order=data.flags.c_contiguous,
    mode=which_mode(tolerance, rate, precision),
    correlated_dims=correlated_dims,
  )

  orderfn = np.asfortranarray if data.flags.f_contiguous else np.ascontiguousarray
  order = ('C' if data.flags.c_contiguous else 'F')

  streams = []
  for slc in compute_slices(header):
    hyperplane = orderfn(data[slc])
    streams.append(
      zfpy.compress_numpy(
        hyperplane,
        tolerance=tolerance ,
        precision=precision,
        rate=rate,
        write_header=True,
      )
    )

  return assemble_container(header, streams)

def assemble_container(header:ZfpcHeader, streams:List[bytes]) -> bytes:
  """streams should be sorted in Fortran order."""
  index_size = (1 + len(streams)) * 8
  index_offset = ZfpcHeader.header_size + index_size
  index_offset = int.to_bytes(index_offset, 8, byteorder="little", signed=False) 

  index_stream = [
    int.to_bytes(len(stream), 8, byteorder="little", signed=False)
    for stream in streams
  ]
  index_stream = [ index_offset ] + index_stream

  buffers = [
    header.tobytes(),
    b''.join(index_stream),
    b''.join(streams)
  ]

  return b''.join(buffers)

def decompress(binary:bytes) -> np.ndarray:
  """
  Decompresses a zfpc file into a numpy array.
  """
  header = ZfpcHeader.frombytes(binary)
  streams = disassemble_container(header, binary)

  order = 'C' if header.c_order else 'F'

  shape = [ header.nx, header.ny, header.nz, header.nw ]
  while len(shape) and shape[-1] == 0:
    shape.pop()

  image = np.zeros(shape, dtype=np.float32, order=order)

  for slc, stream in zip(compute_slices(header), streams):
    image[slc] = zfpy.decompress_numpy(stream)

  return image

def disassemble_container(header:ZfpcHeader, binary:bytes) -> List[bytes]:
  index_bytes = (1 + header.num_streams()) * 8
  index = binary[ZfpcHeader.header_size:ZfpcHeader.header_size + index_bytes]
  index = list(np.frombuffer(index, dtype=np.uint64))

  for size in index:
    assert size < len(binary), f"Index {size} larger than size of buffer ({len(binary)})."

  offset = int(index[0])
  sizes = index[1:]

  streams = []
  for size in map(int, sizes):
    streams.append(
      binary[offset:offset+size]
    )
    offset += size

  return streams

