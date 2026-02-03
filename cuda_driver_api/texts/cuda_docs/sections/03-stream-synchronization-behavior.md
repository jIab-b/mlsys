# Stream Synchronization Behavior

## 3.Â Stream synchronization behavior

## Default stream

The default stream, used when 0 is passed as a cudaStream_t or by APIs that operate on a stream implicitly, can be configured to have either [legacy](<stream-sync-behavior.html#stream-sync-behavior__legacy-default-stream>) or [per-thread](<stream-sync-behavior.html#stream-sync-behavior__per-thread-default-stream>) synchronization behavior as described below.

The behavior can be controlled per compilation unit with the \--default-stream nvcc option. Alternatively, per-thread behavior can be enabled by defining the CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including any CUDA headers. Either way, the CUDA_API_PER_THREAD_DEFAULT_STREAM macro will be defined in compilation units using per-thread synchronization behavior.

## Legacy default stream

The legacy default stream is an implicit stream which synchronizes with all other streams in the same CUcontext except for non-blocking streams, described below. (For applications using the runtime APIs only, there will be one context per device.) When an action is taken in the legacy stream such as a kernel launch or cudaStreamWaitEvent(), the legacy stream first waits on all blocking streams, the action is queued in the legacy stream, and then all blocking streams wait on the legacy stream.

For example, the following code launches a kernel k_1 in stream s, then k_2 in the legacy stream, then k_3 in stream s:


    k_1<<<1, 1, 0, s>>>();
    k_2<<<1, 1>>>();
    k_3<<<1, 1, 0, s>>>();

The resulting behavior is that k_2 will block on k_1 and k_3 will block on k_2.

Non-blocking streams which do not synchronize with the legacy stream can be created using the cudaStreamNonBlocking flag with the stream creation APIs.

The legacy default stream can be used explicitly with the CUstream (cudaStream_t) handle CU_STREAM_LEGACY (cudaStreamLegacy).

## Per-thread default stream

The per-thread default stream is an implicit stream local to both the thread and the CUcontext, and which does not synchronize with other streams (just like explicitly created streams). The per-thread default stream is not a non-blocking stream and will synchronize with the legacy default stream if both are used in a program.

The per-thread default stream can be used explicitly with the CUstream (cudaStream_t) handle CU_STREAM_PER_THREAD (cudaStreamPerThread).

* * *
