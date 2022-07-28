import numpy as np
import zfpc
import pytest

@pytest.fixture
def vector_img():
	return np.load("vector_field_sample.npy")

@pytest.mark.parametrize("correlated_dims", [[True,True,True,True], [True,True,False,False], [False,True,True,True]])
@pytest.mark.parametrize("order", ['C', 'F'])
@pytest.mark.parametrize("tolerance", [ 0.000002, 0.0001, 0.01 ])
def test_compression_decompression_fixed_accuracy(vector_img, correlated_dims, order, tolerance):
	if order == "C":
		vector_img = np.ascontiguousarray(vector_img)
	else:
		vector_img = np.asfortranarray(vector_img)

	compressed = zfpc.compress(
		vector_img, 
		tolerance=tolerance, 
		correlated_dims=correlated_dims
	)
	recovered = zfpc.decompress(compressed)

	print(f"compression ratio: {len(compressed)/vector_img.nbytes*100:.3f}%")

	assert np.allclose(recovered, vector_img, atol=tolerance)

@pytest.mark.parametrize("correlated_dims", [[True,True,True,True], [True,True,False,False], [False,True,True,True]])
@pytest.mark.parametrize("order", ['C', 'F'])
@pytest.mark.parametrize("rate", [ 8, 10, 12 ])
def test_compression_decompression_fixed_rate(vector_img, correlated_dims, order, rate):
	if order == "C":
		vector_img = np.ascontiguousarray(vector_img)
	else:
		vector_img = np.asfortranarray(vector_img)

	compressed = zfpc.compress(
		vector_img, 
		rate=rate, 
		correlated_dims=correlated_dims
	)
	recovered = zfpc.decompress(compressed)
	print(len(compressed) / vector_img.nbytes)
	print(np.max(np.abs(recovered - vector_img)))

