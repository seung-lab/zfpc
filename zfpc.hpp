/*
zfpc: zfp container

Optimally compressed partially corellated zfp streams container.

zfp doesn't optimally compress multi-channel data that
are not well correlated with each other. zfpc splits the
correlated data into different compressed streams and 
serializes the streams into a single file. You can then
treat the multiple compressed streams as a single compressed
file (including random access).

https://zfp.readthedocs.io/en/latest/faq.html#q-vfields

License: LGPLv3+
Author: William Silversmith
Affiliation: Princeton Neuroscience Institute
Date: July 2022
*/

#ifndef __ZFPC_HPP__
#define __ZFPC_HPP__

namespace zfpc {

// little endian serialization of integers to chars
// returns bytes written
inline size_t itoc(uint8_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx] = x;
	return 1;
}

inline size_t itoc(uint16_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	return 2;
}

inline size_t itoc(uint32_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	return 4;
}

inline size_t itoc(uint64_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	buf[idx + 4] = (x >> 32) & 0xFF;
	buf[idx + 5] = (x >> 40) & 0xFF;
	buf[idx + 6] = (x >> 48) & 0xFF;
	buf[idx + 7] = (x >> 56) & 0xFF;
	return 8;
}

template <typename T>
T ctoi(unsigned char* buf, size_t idx = 0);

template <>
uint64_t ctoi(unsigned char* buf, size_t idx) {
	uint64_t x = 0;
	x += static_cast<uint64_t>(buf[idx + 0]) << 0;
	x += static_cast<uint64_t>(buf[idx + 1]) << 8;
	x += static_cast<uint64_t>(buf[idx + 2]) << 16;
	x += static_cast<uint64_t>(buf[idx + 3]) << 24;
	x += static_cast<uint64_t>(buf[idx + 4]) << 32;
	x += static_cast<uint64_t>(buf[idx + 5]) << 40;
	x += static_cast<uint64_t>(buf[idx + 6]) << 48;
	x += static_cast<uint64_t>(buf[idx + 7]) << 56;
	return x;
}

template <>
uint32_t ctoi(unsigned char* buf, size_t idx) {
	uint32_t x = 0;
	x += static_cast<uint32_t>(buf[idx + 0]) << 0;
	x += static_cast<uint32_t>(buf[idx + 1]) << 8;
	x += static_cast<uint32_t>(buf[idx + 2]) << 16;
	x += static_cast<uint32_t>(buf[idx + 3]) << 24;
	return x;
}

template <>
uint16_t ctoi(unsigned char* buf, size_t idx) {
	uint16_t x = 0;
	x += static_cast<uint16_t>(buf[idx + 0]) << 0;
	x += static_cast<uint16_t>(buf[idx + 1]) << 8;
	return x;
}

template <>
uint8_t ctoi(unsigned char* buf, size_t idx) {
	return static_cast<uint8_t>(buf[idx]);
}


/* Header: 
 *   'zfpc'            : magic number (4 bytes)
 *   format version    : unsigned integer (1 byte) 
 *   data type        : unsigned integer (1 byte)
 *   nx, ny, nz, nw   : size of each dimension (2 bytes x4)
 *   correlated_dims  : bitfield (least significant 4 bits (nibble)) (1 byte)
 */
struct ZfpcHeader {
public:
	static constexpr size_t header_size{15};

	static constexpr char magic[4]{ 'z', 'f', 'p', 'c' }; 
	uint8_t format_version; // 0: no z index ; 1: with z index
	uint8_t data_type; // label width in bits
	uint16_t nx;
	uint16_t ny;
	uint16_t nz;
	uint16_t nw;
	uint8_t correlated_dims;

	ZfpcHeader() :
		format_version(0), data_type(0), 
		nx(1), ny(1), nz(1), nw(1),
		correlated_dims(0b1111)
	{}

	ZfpcHeader(
		const uint8_t _format_version, const uint8_t _data_type,
		const uint16_t _nx, const uint16_t _ny, 
		const uint16_t _nz, const uint16_t _nw,
		const uint8_t _correlated_dims
	) : 
		format_version(_format_version), data_type(_data_type), 
		nx(_nx), ny(_ny), nz(_nz), nw(_nw),
		correlated_dims(_correlated_dims)
	{}

	ZfpcHeader(unsigned char* buf, const size_t buflen) {
		if (buflen < header_size) {
			throw std::runtime_error("zfpc: Data stream is not valid. Too short, unable to decompress.");
		}

		bool valid_magic = (buf[0] == 'z' && buf[1] == 'f' && buf[2] == 'p' && buf[3] == 'c');
		format_version = buf[4];

		if (!valid_magic || format_version > 0) {
			throw std::runtime_error("zfpc: Data stream is not valid. Unable to decompress.");
		}

		data_type = ctoi<uint8_t>(buf, 5);
		nx = ctoi<uint16_t>(buf, 6); 
		ny = ctoi<uint16_t>(buf, 8); 
		nz = ctoi<uint16_t>(buf, 10);
		nw = ctoi<uint16_t>(buf, 12);
		correlated_dims = ctoi<uint8_t>(buf, 14);
		
		if (data_type > 4) {
			std::string err = "zfpc: Invalid data type in stream. Unable to decompress. Got: ";
			err += std::to_string(data_width);
			throw std::runtime_error(err);
		}
	}

	size_t tochars(std::vector<unsigned char> &buf, size_t idx = 0) const {
		if ((idx + CompressoHeader::header_size) > buf.size()) {
			throw std::runtime_error("zfpc: Unable to write past end of buffer.");
		}

		size_t i = idx;
		for (int j = 0; j < 4; j++, i++) {
			buf[i] = magic[j];
		}

		i += itoc(format_version, buf, i);
		i += itoc(data_type, buf, i);
		i += itoc(nx, buf, i);
		i += itoc(ny, buf, i);
		i += itoc(nz, buf, i);
		i += itoc(nw, buf, i);
		i += itoc(correlated_dims, buf, i);

		return i - idx;
	}

	static bool valid_header(unsigned char* buf) {
		bool valid_magic = (buf[0] == 'c' && buf[1] == 'p' && buf[2] == 's' && buf[3] == 'o');
		uint8_t format_version = buf[4];
		uint8_t dwidth = ctoi<uint8_t>(buf, 5);
		uint8_t connect = ctoi<uint8_t>(buf, 35);

		bool valid_dtype = (dwidth == 1 || dwidth == 2 || dwidth == 4 || dwidth == 8);
		bool valid_connectivity = (connect == 4 || connect == 6);

		return valid_magic && (format_version < 2) && valid_dtype && valid_connectivity;
	}

	static ZfpcHeader fromchars(unsigned char* buf, const size_t buflen) {
		return ZfpcHeader(buf, buflen);
	}
};





};

#endif