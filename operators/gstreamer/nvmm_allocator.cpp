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

#include "nvmm_allocator.hpp"

#if HOLOSCAN_GSTREAMER_NVMM_SUPPORT

#include <cstring>

#include <holoscan/logger/logger.hpp>
#include <nvbufsurface.h>

namespace holoscan {

NvmmAllocator::~NvmmAllocator() {
  // Destroy any remaining NvBufSurface objects that were not freed
  std::lock_guard<std::mutex> lock(map_mutex_);
  for (auto& [ptr, surface] : surface_map_) {
    if (surface) {
      HOLOSCAN_LOG_WARN("NvmmAllocator: destroying leaked NvBufSurface at {}", ptr);
      NvBufSurfaceDestroy(surface);
    }
  }
  surface_map_.clear();
}

void NvmmAllocator::setup(ComponentSpec& spec) {
  spec.param(width_,
             "width",
             "Width",
             "Frame width in pixels",
             static_cast<int32_t>(1920));
  spec.param(height_,
             "height",
             "Height",
             "Frame height in pixels",
             static_cast<int32_t>(1080));
  spec.param(gpu_id_,
             "gpu_id",
             "GPU ID",
             "CUDA GPU device ID for NVMM allocations",
             static_cast<int32_t>(0));
  spec.param(color_format_,
             "color_format",
             "Color Format",
             "NvBufSurfaceColorFormat enum value (e.g., 4 = NVBUF_COLOR_FORMAT_RGBA)",
             static_cast<int32_t>(4));  // NVBUF_COLOR_FORMAT_RGBA
  spec.param(mem_type_,
             "mem_type",
             "Memory Type",
             "NvBufSurfaceMemType enum value (e.g., 2 = NVBUF_MEM_CUDA_DEVICE)",
             static_cast<int32_t>(2));  // NVBUF_MEM_CUDA_DEVICE
}

void NvmmAllocator::initialize() {
  Allocator::initialize();

  HOLOSCAN_LOG_INFO("NvmmAllocator: initializing with {}x{}, gpu_id={}, color_format={}, "
                    "mem_type={}",
                    width_.get(), height_.get(), gpu_id_.get(),
                    color_format_.get(), mem_type_.get());

  // Validate parameters
  if (width_.get() <= 0 || height_.get() <= 0) {
    throw std::runtime_error("NvmmAllocator: invalid dimensions " +
                             std::to_string(width_.get()) + "x" +
                             std::to_string(height_.get()));
  }

  if (gpu_id_.get() < 0) {
    throw std::runtime_error("NvmmAllocator: invalid GPU ID " +
                             std::to_string(gpu_id_.get()));
  }

  // Validate by doing a test allocation and immediate free
  NvBufSurfaceCreateParams create_params;
  std::memset(&create_params, 0, sizeof(create_params));
  create_params.gpuId = static_cast<uint32_t>(gpu_id_.get());
  create_params.width = static_cast<uint32_t>(width_.get());
  create_params.height = static_cast<uint32_t>(height_.get());
  create_params.size = 0;  // Auto-calculated from width/height/format
  create_params.colorFormat = static_cast<NvBufSurfaceColorFormat>(color_format_.get());
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = static_cast<NvBufSurfaceMemType>(mem_type_.get());
  create_params.isContiguous = true;

  NvBufSurface* test_surface = nullptr;
  int ret = NvBufSurfaceCreate(&test_surface, 1, &create_params);
  if (ret != 0 || !test_surface) {
    throw std::runtime_error("NvmmAllocator: NvBufSurfaceCreate validation failed (ret=" +
                             std::to_string(ret) + ")");
  }

  HOLOSCAN_LOG_INFO("NvmmAllocator: validation succeeded - surface pitch={}, size={}",
                    test_surface->surfaceList[0].pitch,
                    test_surface->surfaceList[0].dataSize);

  NvBufSurfaceDestroy(test_surface);

  initialized_ = true;
  HOLOSCAN_LOG_INFO("NvmmAllocator: initialized successfully");
}

bool NvmmAllocator::is_available(uint64_t size) {
  return initialized_;
}

nvidia::byte* NvmmAllocator::allocate(uint64_t size, nvidia::gxf::MemoryStorageType type) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("NvmmAllocator: not initialized");
    return nullptr;
  }

  // NvBufSurface only supports device memory on dGPU
  if (type != nvidia::gxf::MemoryStorageType::kDevice &&
      type != nvidia::gxf::MemoryStorageType::kSystem) {
    // Accept kSystem as well since the allocator interface may pass it
    // We always allocate on device regardless
    HOLOSCAN_LOG_DEBUG("NvmmAllocator: requested storage type {} will be allocated as NVMM device "
                       "memory",
                       static_cast<int>(type));
  }

  NvBufSurfaceCreateParams create_params;
  std::memset(&create_params, 0, sizeof(create_params));
  create_params.gpuId = static_cast<uint32_t>(gpu_id_.get());
  create_params.width = static_cast<uint32_t>(width_.get());
  create_params.height = static_cast<uint32_t>(height_.get());
  create_params.size = 0;  // Auto-calculated
  create_params.colorFormat = static_cast<NvBufSurfaceColorFormat>(color_format_.get());
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = static_cast<NvBufSurfaceMemType>(mem_type_.get());
  create_params.isContiguous = true;

  NvBufSurface* surface = nullptr;
  int ret = NvBufSurfaceCreate(&surface, 1, &create_params);
  if (ret != 0 || !surface) {
    HOLOSCAN_LOG_ERROR("NvmmAllocator: NvBufSurfaceCreate failed (ret={})", ret);
    return nullptr;
  }

  // Mark the surface as filled
  surface->numFilled = 1;

  // Get the GPU data pointer
  void* data_ptr = surface->surfaceList[0].dataPtr;
  if (!data_ptr) {
    HOLOSCAN_LOG_ERROR("NvmmAllocator: NvBufSurface dataPtr is null after creation");
    NvBufSurfaceDestroy(surface);
    return nullptr;
  }

  // Store the mapping
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    surface_map_[data_ptr] = surface;
  }

  HOLOSCAN_LOG_DEBUG("NvmmAllocator: allocated NvBufSurface at {} (dataPtr={}, size={})",
                     static_cast<void*>(surface), data_ptr,
                     surface->surfaceList[0].dataSize);

  return static_cast<nvidia::byte*>(data_ptr);
}

void NvmmAllocator::free(nvidia::byte* pointer) {
  if (!pointer) {
    return;
  }

  NvBufSurface* surface = nullptr;
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = surface_map_.find(static_cast<void*>(pointer));
    if (it != surface_map_.end()) {
      surface = it->second;
      surface_map_.erase(it);
    }
  }

  if (surface) {
    HOLOSCAN_LOG_DEBUG("NvmmAllocator: freeing NvBufSurface at {} (dataPtr={})",
                       static_cast<void*>(surface), static_cast<void*>(pointer));
    NvBufSurfaceDestroy(surface);
  } else {
    HOLOSCAN_LOG_WARN("NvmmAllocator: free() called with unknown pointer {}", 
                      static_cast<void*>(pointer));
  }
}

NvBufSurface* NvmmAllocator::lookup_surface(void* data_ptr) const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = surface_map_.find(data_ptr);
  if (it != surface_map_.end()) {
    return it->second;
  }
  return nullptr;
}

NvBufSurface* NvmmAllocator::detach_surface(void* data_ptr) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = surface_map_.find(data_ptr);
  if (it != surface_map_.end()) {
    NvBufSurface* surface = it->second;
    surface_map_.erase(it);
    return surface;
  }
  return nullptr;
}

}  // namespace holoscan

#endif  // HOLOSCAN_GSTREAMER_NVMM_SUPPORT
