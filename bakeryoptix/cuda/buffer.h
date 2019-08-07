/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <cuda/preprocessor.h>

enum PageLockedState
{
	UNLOCKED,
	LOCKED
};


// A simple abstraction for memory to be passed into Prime via BufferDescs
template <typename T>
class Buffer
{
public:
	Buffer(const size_t count = 0, const RTPbuffertype type = RTP_BUFFER_TYPE_HOST,
		const PageLockedState page_locked_state = UNLOCKED, const unsigned stride = 0)
		: ptr_(nullptr), page_locked_state_(page_locked_state), temp_host_(nullptr)
	{
		alloc(count, type, page_locked_state, stride);
	}

	// Allocate without changing type or stride
	void alloc(const size_t count)
	{
		alloc(count, type_, page_locked_state_);
	}

	void alloc(const size_t count, const RTPbuffertype type, const PageLockedState page_locked_state = UNLOCKED, const unsigned stride = 0)
	{
		if (ptr_) free();

		type_ = type;
		count_ = count;
		stride_ = stride;

		if (count_ > 0)
		{
			if (type_ == RTP_BUFFER_TYPE_HOST)
			{
				ptr_ = (T*)malloc(size_in_bytes());
				if (page_locked_state) rtpHostBufferLock(ptr_, size_in_bytes()); // for improved transfer performance
				page_locked_state_ = page_locked_state;
			}
			else
			{
				CHK_CUDA( cudaGetDevice( &device_ ) );
				CHK_CUDA( cudaMalloc( &ptr_, size_in_bytes() ) );
			}
		}
	}

	void free()
	{
		if (ptr_ && type_ == RTP_BUFFER_TYPE_HOST)
		{
			if (page_locked_state_)
			{
				rtpHostBufferUnlock(ptr_);
			}
			::free(ptr_);
			::free(temp_host_);
		}
		else
		{
			int old_device;
			CHK_CUDA( cudaGetDevice( &old_device ) );
			CHK_CUDA( cudaSetDevice( device_ ) );
			CHK_CUDA( cudaFree( ptr_ ) );
			CHK_CUDA( cudaSetDevice( old_device ) );
		}

		ptr_ = 0;
		temp_host_ = 0;
		count_ = 0;
		stride_ = 0;
	}

	~Buffer()
	{
		free();
	}

	size_t count() const { return count_; }
	size_t size_in_bytes() const { return count_ * (stride_ ? stride_ : sizeof(T)); }
	const T* ptr() const { return ptr_; }
	T* ptr() { return ptr_; }
	RTPbuffertype type() const { return type_; }
	unsigned stride() const { return stride_; }

	const T* host_ptr()
	{
		if (type_ == RTP_BUFFER_TYPE_HOST) return ptr_;

		if (!temp_host_) temp_host_ = (T*)malloc(size_in_bytes());
		CHK_CUDA( cudaMemcpy( &temp_host_[0], ptr_, size_in_bytes(), cudaMemcpyDeviceToHost ) );
		return &temp_host_[0];
	}

protected:
	RTPbuffertype type_;
	T* ptr_;
	int device_;
	size_t count_;
	unsigned stride_;
	PageLockedState page_locked_state_;
	T* temp_host_;

public:
	Buffer<T>(const Buffer<T>&) = delete; // forbidden
	Buffer<T>& operator=(const Buffer<T>&) = delete; // forbidden
};
