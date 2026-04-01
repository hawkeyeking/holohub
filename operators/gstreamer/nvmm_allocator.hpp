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

#ifndef HOLOSCAN__GSTREAMER__NVMM_ALLOCATOR_HPP
#define HOLOSCAN__GSTREAMER__NVMM_ALLOCATOR_HPP

#include "gst/config.hpp"

#if HOLOSCAN_GSTREAMER_NVMM_SUPPORT

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <holoscan/holoscan.hpp>

// Forward declaration - NvBufSurface is defined in DeepStream SDK headers
struct NvBufSurface;

namespace holoscan {

/**
 * @brief Holoscan allocator backed by NvBufSurface (NVMM) memory
 *
 * This allocator creates NvBufSurface objects and returns their GPU data pointers.
 * It maintains a mapping from data pointer → NvBufSurface* so that downstream
 * components (e.g., NvmmMemoryWrapper in GstSrcBridge) can retrieve the original
 * NvBufSurface for zero-copy GstBuffer wrapping.
 *
 * Key design:
 * - allocate() creates an NvBufSurface via NvBufSurfaceCreate() and returns dataPtr
 * - free() calls NvBufSurfaceDestroy() and removes the mapping
 * - lookup_surface() allows external code to find the NvBufSurface* from a data pointer
 * - Thread-safe via mutex protection on the pointer map
 *
 * Usage:
 * 1. Configure with width, height, color format, GPU ID, memory type
 * 2. Pass as allocator to Holoscan operators (e.g., PatternGenOperator)
 * 3. The bridge's NvmmMemoryWrapper calls lookup_surface() to get NvBufSurface*
 * 4. NvBufSurface lifetime extends until GstBuffer is freed (via separate ref tracking)
 */
class NvmmAllocator : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(NvmmAllocator)

  NvmmAllocator() = default;
  ~NvmmAllocator() override;

  // Non-copyable and non-movable
  NvmmAllocator(const NvmmAllocator&) = delete;
  NvmmAllocator& operator=(const NvmmAllocator&) = delete;
  NvmmAllocator(NvmmAllocator&&) = delete;
  NvmmAllocator& operator=(NvmmAllocator&&) = delete;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  /**
   * @brief Check if the allocator is available (always true after initialize)
   */
  bool is_available(uint64_t size) override;

  /**
   * @brief Allocate NVMM-backed memory
   *
   * Creates an NvBufSurface and returns its GPU data pointer. The size parameter
   * is used for validation but the actual allocation size is determined by the
   * configured width, height, and color format.
   *
   * @param size Requested size in bytes (used for validation)
   * @param type Memory storage type (only kDevice is supported)
   * @return GPU data pointer from the NvBufSurface, or nullptr on failure
   */
  nvidia::byte* allocate(uint64_t size, nvidia::gxf::MemoryStorageType type) override;

  /**
   * @brief Free NVMM-backed memory
   *
   * Destroys the NvBufSurface associated with the given pointer and removes
   * it from the internal mapping.
   *
   * @param pointer Pointer previously returned by allocate()
   */
  void free(nvidia::byte* pointer) override;

  /**
   * @brief Look up the NvBufSurface associated with a data pointer
   *
   * This is the key integration point for NvmmMemoryWrapper: given a tensor's
   * data pointer (which was allocated by this allocator), retrieve the original
   * NvBufSurface* for zero-copy GstBuffer wrapping.
   *
   * @param data_ptr GPU data pointer (from tensor->data())
   * @return NvBufSurface* if found, nullptr otherwise
   */
  NvBufSurface* lookup_surface(void* data_ptr) const;

  /**
   * @brief Remove a surface from the mapping without destroying it
   *
   * Used when ownership of the NvBufSurface is transferred to a GstBuffer.
   * The GstBuffer's destroy_notify will call NvBufSurfaceDestroy when it's done.
   *
   * @param data_ptr GPU data pointer to remove from the mapping
   * @return NvBufSurface* if found and removed, nullptr otherwise
   */
  NvBufSurface* detach_surface(void* data_ptr);

 private:
  // Configuration parameters
  Parameter<int32_t> width_;
  Parameter<int32_t> height_;
  Parameter<int32_t> gpu_id_;
  Parameter<int32_t> color_format_;  // NvBufSurfaceColorFormat as int
  Parameter<int32_t> mem_type_;      // NvBufSurfaceMemType as int

  // Thread-safe mapping from GPU data pointer → NvBufSurface*
  mutable std::mutex map_mutex_;
  std::unordered_map<void*, NvBufSurface*> surface_map_;

  bool initialized_ = false;
};

}  // namespace holoscan

#endif  // HOLOSCAN_GSTREAMER_NVMM_SUPPORT

#endif /* HOLOSCAN__GSTREAMER__NVMM_ALLOCATOR_HPP */
