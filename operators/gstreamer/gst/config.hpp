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

#ifndef HOLOSCAN__GSTREAMER__GST__CONFIG_HPP
#define HOLOSCAN__GSTREAMER__GST__CONFIG_HPP

#include <gst/gstversion.h>

// CUDA support in GStreamer requires version 1.24+.
// This includes functions like gst_cuda_memory_init_once() and gst_cuda_allocator_alloc_wrapped().
// Enable CUDA support only if GStreamer version is 1.24.0 or higher.
#if GST_CHECK_VERSION(1, 24, 0)
#define HOLOSCAN_GSTREAMER_CUDA_SUPPORT 1
#else
#define HOLOSCAN_GSTREAMER_CUDA_SUPPORT 0
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

// NVMM (NvBufSurface) support for DeepStream integration.
// Defined by CMakeLists.txt when the DeepStream SDK nvbufsurface library is found.
// When enabled, provides NvmmAllocator and NvmmMemoryWrapper for zero-copy
// Holoscan-to-DeepStream data flow via memory:NVMM caps.
#ifndef HOLOSCAN_GSTREAMER_NVMM_SUPPORT
#define HOLOSCAN_GSTREAMER_NVMM_SUPPORT 0
#endif  // HOLOSCAN_GSTREAMER_NVMM_SUPPORT

#endif /* GST_CONFIG_HPP */
