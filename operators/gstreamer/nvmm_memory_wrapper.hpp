/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN__GSTREAMER__NVMM_MEMORY_WRAPPER_HPP
#define HOLOSCAN__GSTREAMER__NVMM_MEMORY_WRAPPER_HPP

#include "gst/config.hpp"

#if HOLOSCAN_GSTREAMER_NVMM_SUPPORT

#include <gst/gst.h>

#include <cstring>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/logger/logger.hpp>
#include <nvbufsurface.h>

#include "gst/buffer.hpp"
#include "gst/memory.hpp"
#include "gst_src_bridge.hpp"
#include "nvmm_allocator.hpp"

namespace holoscan {

/**
 * @brief NVMM memory wrapper for zero-copy GstBuffer creation from NvBufSurface-backed tensors
 *
 * This wrapper is used when the GStreamer caps specify `memory:NVMM`. It retrieves the
 * NvBufSurface* from the NvmmAllocator's internal mapping (given the tensor's data pointer)
 * and wraps it into a GstBuffer that DeepStream plugins can consume directly.
 *
 * The wrapping approach:
 * 1. Tensor's data pointer → NvmmAllocator::lookup_surface() → NvBufSurface*
 * 2. Detach the surface from the allocator (ownership transferred to GstBuffer)
 * 3. Create a GstMemory that wraps the NvBufSurface's dataPtr
 * 4. Attach the NvBufSurface* as GstMeta or use destroy_notify for lifecycle
 * 5. The GstBuffer destruction triggers NvBufSurfaceDestroy via destroy_notify
 *
 * For DeepStream compatibility, the GstBuffer must:
 * - Have caps with `video/x-raw(memory:NVMM)`
 * - Contain memory whose data pointer is the NvBufSurface's surfaceList[0].dataPtr
 * - Have the NvBufSurface properly accessible for DeepStream metadata attachment
 */
class NvmmMemoryWrapper : public GstSrcBridge::MemoryWrapper {
 public:
  /**
   * @brief Construct NVMM memory wrapper
   * @param video_format GStreamer video format from caps
   * @param allocator The NvmmAllocator that created the surfaces (for lookup/detach)
   */
  NvmmMemoryWrapper(GstVideoFormat video_format, std::shared_ptr<NvmmAllocator> allocator)
      : MemoryWrapper(video_format), nvmm_allocator_(std::move(allocator)) {
    HOLOSCAN_LOG_INFO("Initializing NVMM memory wrapper for DeepStream zero-copy integration");
  }

  bool validate(const holoscan::Tensor* tensor) const override {
    // Check tensor validity
    if (!tensor->data() || tensor->nbytes() == 0) {
      HOLOSCAN_LOG_ERROR("Invalid tensor data or size for NVMM memory wrapping");
      return false;
    }

    // NVMM allocator produces device memory, so tensor should be on CUDA device
    DLDevice device = tensor->device();
    if (device.device_type != kDLCUDA && device.device_type != kDLCUDAManaged) {
      HOLOSCAN_LOG_ERROR(
          "NvmmMemoryWrapper expects GPU memory (kDLCUDA or kDLCUDAManaged), but tensor is on "
          "device type {}. Ensure the NvmmAllocator is being used.",
          static_cast<int>(device.device_type));
      return false;
    }

    // Verify the tensor's data pointer is known to our allocator
    if (!nvmm_allocator_) {
      HOLOSCAN_LOG_ERROR("NvmmMemoryWrapper: NvmmAllocator is null");
      return false;
    }

    NvBufSurface* surface = nvmm_allocator_->lookup_surface(tensor->data());
    if (!surface) {
      HOLOSCAN_LOG_ERROR(
          "NvmmMemoryWrapper: tensor data pointer {} not found in NvmmAllocator's surface map. "
          "Ensure the tensor was allocated by the NvmmAllocator.",
          tensor->data());
      return false;
    }

    return true;
  }

  gst::Memory wrap_memory(const holoscan::Tensor* tensor, void* user_data,
                          ::GDestroyNotify destroy_notify) override {
    void* tensor_data = tensor->data();
    size_t tensor_size = tensor->nbytes();

    // Look up the NvBufSurface from the allocator
    NvBufSurface* surface = nvmm_allocator_->lookup_surface(tensor_data);
    if (!surface) {
      HOLOSCAN_LOG_ERROR("NvmmMemoryWrapper: cannot find NvBufSurface for data pointer {}",
                         tensor_data);
      return gst::Memory();
    }

    HOLOSCAN_LOG_DEBUG("NvmmMemoryWrapper: wrapping NvBufSurface at {} (dataPtr={}, size={})",
                       static_cast<void*>(surface), tensor_data,
                       surface->surfaceList[0].dataSize);

    // Create a combined cleanup context that:
    // 1. Calls the original destroy_notify (to release the Holoscan tensor reference)
    // 2. Does NOT destroy the NvBufSurface here - it stays owned by the allocator
    //    and will be freed when the allocator's free() is called by the GXF entity destruction
    //
    // The key insight: the NvBufSurface lifetime is managed by the allocator, which is
    // managed by the Holoscan entity. The GstBuffer just wraps the pointer. The tensor's
    // DLManagedTensorContext (held by user_data / destroy_notify) keeps the GXF entity alive,
    // which keeps the allocator's mapping valid. When the GstBuffer is freed, destroy_notify
    // releases the tensor reference, which may trigger GXF entity destruction, which calls
    // allocator->free() on the data pointer.

    // The memory size should be the NvBufSurface's actual allocation size, not the tensor size,
    // because NvBufSurface may have padding (pitch alignment).
    size_t surface_size = surface->surfaceList[0].dataSize;

    // Wrap the NvBufSurface's data pointer as GstMemory
    // We use the surface's dataSize as the actual memory size since it includes pitch padding
    gst::Memory memory = gst::Memory::create_wrapped(
        static_cast<GstMemoryFlags>(0),  // flags (read-write)
        tensor_data,                     // data pointer (NvBufSurface's GPU dataPtr)
        surface_size,                    // maxsize (actual surface allocation size)
        0,                               // offset
        surface_size,                    // size (use full surface allocation)
        user_data,                       // user_data (TensorWrapper to keep tensor alive)
        destroy_notify);                 // destroy_notify (frees TensorWrapper)

    if (!memory) {
      HOLOSCAN_LOG_ERROR("NvmmMemoryWrapper: failed to create wrapped GstMemory");
      return gst::Memory();
    }

    HOLOSCAN_LOG_DEBUG("NvmmMemoryWrapper: successfully created zero-copy NVMM GstMemory "
                       "(surface_size={}, tensor_size={})",
                       surface_size, tensor_size);

    return memory;
  }

 private:
  std::shared_ptr<NvmmAllocator> nvmm_allocator_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_GSTREAMER_NVMM_SUPPORT

#endif /* HOLOSCAN__GSTREAMER__NVMM_MEMORY_WRAPPER_HPP */
