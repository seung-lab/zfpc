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

#include <cstdint>
#include "zfp.h"

namespace zfpc {

// little endian serialization of integers to chars
// returns bytes written
inline uint64_t itoc(uint8_t x, std::vector<unsigned char> &buf, uint64_t idx) {
	buf[idx] = x;
	return 1;
}

inline uint64_t itoc(uint16_t x, std::vector<unsigned char> &buf, uint64_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	return 2;
}

inline uint64_t itoc(uint32_t x, std::vector<unsigned char> &buf, uint64_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	return 4;
}

inline uint64_t itoc(uint64_t x, std::vector<unsigned char> &buf, uint64_t idx) {
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
T ctoi(unsigned char* buf, uint64_t idx = 0);

template <>
uint64_t ctoi(unsigned char* buf, uint64_t idx) {
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
uint32_t ctoi(unsigned char* buf, uint64_t idx) {
	uint32_t x = 0;
	x += static_cast<uint32_t>(buf[idx + 0]) << 0;
	x += static_cast<uint32_t>(buf[idx + 1]) << 8;
	x += static_cast<uint32_t>(buf[idx + 2]) << 16;
	x += static_cast<uint32_t>(buf[idx + 3]) << 24;
	return x;
}

template <>
uint16_t ctoi(unsigned char* buf, uint64_t idx) {
	uint16_t x = 0;
	x += static_cast<uint16_t>(buf[idx + 0]) << 0;
	x += static_cast<uint16_t>(buf[idx + 1]) << 8;
	return x;
}

template <>
uint8_t ctoi(unsigned char* buf, uint64_t idx) {
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
	static constexpr uint64_t header_size{15};

	static constexpr char magic[4]{ 'z', 'f', 'p', 'c' }; 
	uint8_t format_version; // 0: no z index ; 1: with z index
	uint8_t data_type; // label width in bits
	uint16_t nx;
	uint16_t ny;
	uint16_t nz;
	uint16_t nw;
	uint8_t correlated_dims;
	bool c_order;

	ZfpcHeader() :
		format_version(0), data_type(0), 
		nx(1), ny(1), nz(1), nw(1),
		correlated_dims(0b1111), c_order(false)
	{}

	ZfpcHeader(
		const uint8_t _format_version, const uint8_t _data_type,
		const uint16_t _nx, const uint16_t _ny, 
		const uint16_t _nz, const uint16_t _nw,
		const uint8_t _correlated_dims,
		const bool _c_order
	) : 
		format_version(_format_version), data_type(_data_type), 
		nx(_nx), ny(_ny), nz(_nz), nw(_nw),
		correlated_dims(_correlated_dims),
		c_order(_c_order)
	{}

	ZfpcHeader(unsigned char* buf, const uint64_t buflen) {
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

		c_order = (data_type >> 7);
		data_type = data_type & 0b01111111;
		
		if (data_type > 4) {
			std::string err = "zfpc: Invalid data type in stream. Unable to decompress. Got: ";
			err += std::to_string(data_width);
			throw std::runtime_error(err);
		}
	}

	uint64_t tochars(std::vector<unsigned char> &buf, uint64_t idx = 0) const {
		if ((idx + CompressoHeader::header_size) > buf.size()) {
			throw std::runtime_error("zfpc: Unable to write past end of buffer.");
		}

		uint64_t i = idx;
		for (int j = 0; j < 4; j++, i++) {
			buf[i] = magic[j];
		}

		i += itoc(format_version, buf, i);
		i += itoc(data_type | (c_order << 7), buf, i);
		i += itoc(nx, buf, i);
		i += itoc(ny, buf, i);
		i += itoc(nz, buf, i);
		i += itoc(nw, buf, i);
		i += itoc(correlated_dims, buf, i);

		return i - idx;
	}

	static bool valid_header(unsigned char* buf, const uint64_t buflen) {
		if (buflen < header_size) {
			return false;
		}

		bool valid_magic = (buf[0] == 'z' && buf[1] == 'f' && buf[2] == 'p' && buf[3] == 'c');
		uint8_t format_version = buf[4];
		uint8_t dtype = ctoi<uint8_t>(buf, 5);

		bool valid_dtype = (dtype & 0b01111111) < 5;

		return valid_magic && (format_version == 0) && valid_dtype;
	}

	static ZfpcHeader fromchars(unsigned char* buf, const uint64_t buflen) {
		return ZfpcHeader(buf, buflen);
	}
};

std::vector<uint64_t> get_index_offsets() {
	ZfpcHeader header(buf, buflen);
	
	auto streamsfn = [&header, &nstreams](uint8_t i, uint64_t ni) {
		if ((((~header.correlated_dims) >> i) & 0b1) && ni > 1) {
			return nstreams * ni;
		}
		return nstreams;
	};

	uint64_t nstreams = 1;
	nstreams = streamsfn(3, header.nx);
	nstreams = streamsfn(2, header.ny);
	nstreams = streamsfn(1, header.nz);
	nstreams = streamsfn(0, header.nw);

	std::vector<uint64_t> stream_offsets(nstreams);

	uint64_t offset = ZfpcHeader::header_size;

	if (buflen < offset + nstreams * sizeof(uint64_t)) {
		throw std::runtime_error("zfpc: Buffer length too short for stream index.");
	}

	for (int i = 0; i < nstreams; i++, offset += sizeof(uint64_t)) {
		stream_offsets[i] = ctoi<uint64_t>(buf, offset);
	}

	return stream_offsets;
}

uint64_t write_stream(
	const std::vector<unsigned char> &stream, 
	std::vector<unsigned char> &out_stream, 
	uint64_t start_idx
) {
	for (uint64_t i = 0; i < out_stream.size(); i++) {
		stream[start_idx + i] = stream[i];
	}
} 

std::vector<unsigned char> assemble_container(
	const ZfpcHeader &header,
	std::vector<std::vector<unsigned char>> &streams,
) {
	uint64_t total_bytes = ZfpcHeader::header_size;
	uint64_t index_size = sizeof(uint64_t) * streams.size();
	for (auto stream : streams) {
		total_bytes += stream.size();
	}
	total_bytes += index_size;

	std::vector<unsigned char> output(total_bytes);

	uint64_t written = 0;
	header.tochars(output, written);

	// write stream index
	uint64_t index_offset = ZfpcHeader::header_size + index_size;
	uint64_t index_sum = 0;
	for (uint64_t i = 0; i < streams.size(); i++) {
		written += itoc<uint64_t>(index_offset + index_sum, output, written);
		index_sum += streams[i].size();
	}

	// write streams in Fortran order
	for (auto stream : streams) {
		write_stream(stream, output, written);
		written += stream.size();
	}

	return output;
}

uint64_t get_num_streams(const ZfpcHeader &header) {
	uint64_t shape[4] = { header.nx, header.ny, header.nz, header.nw };
	uint64_t num_streams = 1;
	// size 0 is treated as the dimension does not exist. Zeros should
	// only occur on the rhs.
	for (int i = 0; i < 4; i++) {
		if (shape[i] > 1 && ((header.correlated_dims >> i) & 0b1)) {
			num_streams *= shape[i];
		}
	}
	return num_streams;
}

std::vector<uint64_t> read_stream_index(const std::vector<unsigned char> &buf) {
	std::vector<std::vector<unsigned char>> streams;
	ZfpcHeader header(buf.data(), buf.size());
	uint64_t num_streams = get_num_streams(header);

	if (buf.size() < ZfpcHeader::header_size + sizeof(uint64_t) * num_streams) {
		throw std::runtime_error("zfpc: Input stream too short to decode stream index.");
	}

	std::vector<uint64_t> stream_index(num_streams);
	uint64_t offset = ZfpcHeader::header_size;
	for (int i = 0; i < num_streams; i++) {
		stream_index[i] = ctoi<uint64_t>(buf.data(), offset);
		offset += sizeof(uint64_t);

		if (stream_index[i] >= buf.size()) {
			throw std::runtime_error("zfpc: Invalid stream index. Stream location outside of buffer.");
		}
	}

	return stream_index;	
}

std::vector<std::vector<unsigned char>> disassemble_container(
	const std::vector<unsigned char> &buf
) {
	std::vector<uint64_t> stream_index = read_stream_index(buf);


	return output;
}



};

#endif