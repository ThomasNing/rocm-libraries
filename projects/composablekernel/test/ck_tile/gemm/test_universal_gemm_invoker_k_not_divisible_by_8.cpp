// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Validate the gfx1250 vectorized K-tail dispatch through the WMMA example configuration.

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "universal_gemm_invoker.hpp"

#include "ck_tile/host.hpp"

#include "gtest/gtest.h"

#ifdef CK_USE_GFX1250

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename DataType>
class TestUniversalInvokerKNotDivisibleBy8 : public ::testing::Test
{
    protected:
    void RunAndVerify(int K, int k_batch = 1)
    {
        using AccDataType = typename GemmTypeConfig<DataType>::AccDataType;

        constexpr int M = 128;
        constexpr int N = 128;

        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        // Host tensors
        ck_tile::HostTensor<DataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<DataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, ck_tile::bool_constant<false>{}));
        ck_tile::HostTensor<DataType> gpu_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<true>{}));

        ck_tile::FillUniformDistributionIntegerValue<DataType>{-5, 5, 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<DataType>{-5, 5, 11940}(b_k_n);

        // Device buffers
        ck_tile::DeviceMem a_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_buf(gpu_result.get_element_space_size_in_bytes());
        a_buf.ToDevice(a_m_k.data());
        b_buf.ToDevice(b_k_n.data());
        c_buf.SetZero();

        ck_tile::GemmHostArgs args = {a_buf.GetDeviceBuffer(),
                                      b_buf.GetDeviceBuffer(),
                                      c_buf.GetDeviceBuffer(),
                                      k_batch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        // Run UniversalInvoker::gemm()
        UniversalInvoker::gemm<GemmConfigComputeV3_WMMA<DataType>,
                               DataType,
                               DataType,
                               ck_tile::tuple<>,
                               AccDataType,
                               DataType,
                               Row,
                               Col,
                               ck_tile::tuple<>,
                               Row,
                               /*Persistent=*/false,
                               ck_tile::element_wise::PassThrough>(
            args, ck_tile::stream_config{nullptr, false});

        c_buf.FromDevice(gpu_result.data());

        ck_tile::HostTensor<DataType> host_reference(
            ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<true>{}));
        host_reference.SetZero();
        ck_tile::reference_gemm<DataType, DataType, AccDataType, DataType>(
            a_m_k, b_k_n, host_reference);

        const float max_accumulated_value =
            *std::max_element(host_reference.mData.begin(), host_reference.mData.end());
        const auto rtol_atol = calculate_rtol_atol<DataType, DataType, AccDataType, DataType>(
            K, k_batch, max_accumulated_value);

        // Compare both results
        EXPECT_TRUE(do_verify(gpu_result, host_reference, rtol_atol, "GPU"))
            << "K=" << K << ", k_batch=" << k_batch;
    }
};

using KNotDivisibleBy8DataTypes = ::testing::Types<ck_tile::half_t, ck_tile::bf16_t>;
TYPED_TEST_SUITE(TestUniversalInvokerKNotDivisibleBy8, KNotDivisibleBy8DataTypes);

TYPED_TEST(TestUniversalInvokerKNotDivisibleBy8, KNotDivisibleBy8)
{
    // Cover every K%8!=0
    // More values: K = 32+1, 64-3, 128-3, 160-3
    for(int K : {1, 2, 3, 4, 5, 6, 7, 9, 33, 61, 125, 157})
    {
        this->RunAndVerify(K);
    }
}

TYPED_TEST(TestUniversalInvokerKNotDivisibleBy8, AlignedK)
{
    // Aligned K: original path
    this->RunAndVerify(2 * GemmConfigComputeV3_WMMA<TypeParam>::K_Tile);
}

TYPED_TEST(TestUniversalInvokerKNotDivisibleBy8, SplitK)
{
    this->RunAndVerify(/*K=*/37, /*k_batch=*/2);
}

#endif // CK_USE_GFX1250
