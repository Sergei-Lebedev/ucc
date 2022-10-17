/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include <components/ec/ucc_ec.h>
#include <pthread.h>
}
#include <common/test.h>
#include <common/test_ucc.h>
#include <cuda_runtime.h>

class test_ec_cuda : public ucc::test {
    void TestECCudaSetUp(ucc_ec_params_t ec_params)
    {
        ucc::test::SetUp();
        ucc_constructor();
        ucc_ec_init(&ec_params);
    }

    virtual void SetUp() override
    {
        ucc_ec_params_t ec_params = {
            .thread_mode = UCC_THREAD_SINGLE,
        };

        TestECCudaSetUp(ec_params);
        if (UCC_OK != ucc_ec_available(UCC_EE_CUDA_STREAM)) {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() override
    {
        ucc_ec_finalize();
        ucc::test::TearDown();
    }

public:
    ucc_status_t get_cuda_executor(ucc_ee_executor_t **executor)
    {
        ucc_ee_executor_params_t eparams;

        eparams.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
        eparams.ee_type = UCC_EE_CUDA_STREAM;

        return ucc_ee_executor_init(&eparams, executor);
    }

    ucc_status_t put_cuda_executor(ucc_ee_executor_t *executor)
    {
        return ucc_ee_executor_finalize(executor);
    }

};

UCC_TEST_F(test_ec_cuda, ec_cuda_load)
{
    ASSERT_EQ(UCC_OK, ucc_ec_available(UCC_EE_CUDA_STREAM));
}

UCC_TEST_F(test_ec_cuda, ec_cuda_executor_init_finalize)
{
    ucc_ee_executor_t *executor;

    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}

UCC_TEST_F(test_ec_cuda, ec_cuda_executor_interruptible_start)
{
    ucc_ee_executor_t *executor;
    ucc_status_t status;

    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));

    status = ucc_ee_executor_start(executor, nullptr);
    EXPECT_EQ(UCC_OK, status);
    if (status == UCC_OK) {
        EXPECT_EQ(UCC_OK, ucc_ee_executor_stop(executor));
    }
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}


UCC_TEST_F(test_ec_cuda, ec_cuda_executor_interruptible_copy)
{
    ucc_ee_executor_t *executor;
    ucc_status_t status;
    cudaError_t cuda_st;
    int *src_host, *dst_host, *src, *dst;
    const size_t size = 4850;
    ucc_ee_executor_task_t *task;

    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));

    status = ucc_ee_executor_start(executor, nullptr);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    src_host = (int*)malloc(size * sizeof(int));
    EXPECT_NE(nullptr, src_host);
    if (!src_host) {
        goto exit;
    }

    dst_host = (int*)malloc(size * sizeof(int));
    EXPECT_NE(nullptr, dst_host);
    if (!dst_host) {
        goto exit;
    }

    cuda_st = cudaMalloc(&src, size * sizeof(int));
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    cuda_st = cudaMalloc(&dst, size * sizeof(int));
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    for (int i = 0; i < size; i++) {
        src_host[i] = i;
    }

    cuda_st = cudaMemcpy(src, src_host, size * sizeof(int), cudaMemcpyDefault);
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    ucc_ee_executor_task_args_t args;
    args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    args.copy.src  = src;
    args.copy.dst  = dst;
    args.copy.len  = size * sizeof(int);

    status = ucc_ee_executor_task_post(executor, &args, &task);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    do {
        status = ucc_ee_executor_task_test(task);
    } while (status > 0);

    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    cuda_st = cudaMemcpy(dst_host, dst, size * sizeof(int), cudaMemcpyDefault);
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    for (int i = 0; i < size; i++) {
        if (dst_host[i] != src_host[i]) {
            EXPECT_EQ(dst_host[i], src_host[i]);
            goto exit;
        }
    }

exit:
    if (task) {
        EXPECT_EQ(UCC_OK, ucc_ee_executor_task_finalize(task));
    }
    if (src_host) {
        free(src_host);
    }

    if (dst_host) {
        free(dst_host);
    }

    if (src) {
        cudaFree(src);
    }

    if (dst) {
        cudaFree(dst);
    }
    EXPECT_EQ(UCC_OK, ucc_ee_executor_stop(executor));
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}

UCC_TEST_F(test_ec_cuda, ec_cuda_get_compress_size)
{
    size_t count_in, count_out;

    count_in = 8273;
    ASSERT_EQ(ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, count_in,
                                       UCC_DT_FLOAT32, &count_out), UCC_OK);

    ASSERT_GT(count_out, 0);
    ASSERT_EQ(ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, count_in,
                                       UCC_DT_FLOAT16, &count_out), UCC_OK);
    ASSERT_GT(count_out, 0);

    ASSERT_EQ(ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, count_in,
                                       UCC_DT_BFLOAT16, &count_out), UCC_OK);
    ASSERT_GT(count_out, 0);
}

UCC_TEST_F(test_ec_cuda, ec_cuda_executor_interruptible_compress)
{
// test_ec_cuda_ec_cuda_executor_interruptible_compress_Test::TestBody
    ucc_ee_executor_t *executor;
    ucc_status_t status;
    ucc_ee_executor_task_t *task_compress, *task_decompress;
    int size  = 252500;
    int nbufs = 4;
    float *src_dev, *dst_dev;
    float *src_host, *dst_host;
    uint64_t *src_count, *src_displ, *dst_count, *dst_displ;
    size_t max_count = 0;
    size_t comp_size;

    task_compress   = nullptr;
    task_decompress = nullptr;
    src_count = nullptr;
    dst_count = nullptr;
    src_displ = nullptr;
    dst_displ = nullptr;
    src_dev   = nullptr;
    src_host  = nullptr;
    dst_dev   = nullptr;
    dst_host  = nullptr;
    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));

    status = ucc_ee_executor_start(executor, nullptr);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    src_host  = (float*)malloc(size * sizeof(float));
    dst_host  = (float*)malloc(size * sizeof(float));
    src_count = (uint64_t*)malloc(nbufs * sizeof(uint64_t));
    src_displ = (uint64_t*)malloc(nbufs * sizeof(uint64_t));
    dst_count = (uint64_t*)malloc(nbufs * sizeof(uint64_t));
    dst_displ = (uint64_t*)malloc(nbufs * sizeof(uint64_t));

    for (size_t i = 0; i < size; i++) {
        src_host[i] = (float)i;
    }

    src_count[nbufs - 1] = size;
    for (int i = 0; i < nbufs - 1; i++) {
        src_count[i] = (size / nbufs)  - i;
        src_count[nbufs - 1] -= src_count[i];
    }

    src_displ[0] = 0;
    for (int i = 1; i < nbufs; i++) {
        src_displ[i] = src_displ[i - 1] + src_count[i - 1];
    }

    dst_displ[0] = 0;
    for (int i = 1; i < nbufs; i++) {
        ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, src_count[i - 1],
                                 UCC_DT_FLOAT32, &comp_size);
        max_count += comp_size;
        dst_displ[i] = dst_displ[i - 1] + comp_size;
    }
    ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, src_count[nbufs - 1],
                             UCC_DT_FLOAT32, &comp_size);
    max_count += comp_size;

    cudaMalloc(&src_dev, size * sizeof(float));
    cudaMemcpy(src_dev, src_host, size * sizeof(float),
               cudaMemcpyDefault);
    cudaMalloc(&dst_dev, max_count * sizeof(float));

    ucc_ee_executor_task_args_t args;

    args.task_type      = UCC_EE_EXECUTOR_TASK_COMPRESS;
    args.compress.dt    = UCC_DT_FLOAT32;
    args.compress.flags = UCC_EE_TASK_COMPRESS_FLAG_COUNT64 |
                          UCC_EE_TASK_COMPRESS_FLAG_DISPLACEMENT64;
    args.compress.src               = src_dev;
    args.compress.src_counts        = src_count;
    args.compress.src_displacements = src_displ;
    args.compress.dst               = dst_dev;
    args.compress.dst_counts        = dst_count;
    args.compress.dst_displacements = dst_displ;
    args.compress.size              = nbufs;


    status = ucc_ee_executor_task_post(executor, &args, &task_compress);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    do {
        status = ucc_ee_executor_task_test(task_compress);
    } while (status > 0);

    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    cudaMemset(src_dev, 0, size * sizeof(float));

    args.task_type = UCC_EE_EXECUTOR_TASK_COMPRESS;
    args.compress.dt = UCC_DT_FLOAT32;
    args.compress.flags = UCC_EE_TASK_COMPRESS_FLAG_COUNT64 |
                          UCC_EE_TASK_COMPRESS_FLAG_DISPLACEMENT64 |
                          UCC_EE_TASK_COMPRESS_FLAG_DECOMPRESS;
    args.compress.src               = dst_dev;
    args.compress.src_counts        = dst_count;
    args.compress.src_displacements = dst_displ;
    args.compress.dst               = src_dev;
    args.compress.dst_counts        = src_count;
    args.compress.dst_displacements = src_displ;
    args.compress.size              = nbufs;

    status = ucc_ee_executor_task_post(executor, &args, &task_decompress);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    do {
        status = ucc_ee_executor_task_test(task_decompress);
    } while (status > 0);

    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    cudaMemcpy(dst_host, src_dev, size * sizeof(float), cudaMemcpyDefault);

    for (int i = 0; i < size; i++) {
        if (dst_host[i] != src_host[i]) {
            EXPECT_EQ(dst_host[i], src_host[i]);
            goto exit;
        }
    }

exit:
    if (src_host) {
        free(src_host);
    }
    if (dst_host) {
        free(dst_host);
    }
    if (src_dev) {
        cudaFree(src_dev);
    }
    if (dst_dev) {
        cudaFree(dst_dev);
    }
    if (src_count) {
        free(src_count);
    }
    if (dst_count) {
        free(dst_count);
    }
    if (src_displ) {
        free(src_displ);
    }
    if (dst_displ) {
        free(dst_displ);
    }

    if (task_compress) {
        EXPECT_EQ(UCC_OK, ucc_ee_executor_task_finalize(task_compress));
    }
    if (task_decompress) {
        EXPECT_EQ(UCC_OK, ucc_ee_executor_task_finalize(task_decompress));
    }

    EXPECT_EQ(UCC_OK, ucc_ee_executor_stop(executor));
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}
