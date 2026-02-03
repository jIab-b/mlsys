# Coredump Attributes Control

## 6.34.Â Coredump Attributes Control API

This section describes the coredump attribute control functions of the low-level CUDA driver application programming interface.

### Enumerations

enumÂ [CUCoredumpGenerationFlags](<#group__CUDA__COREDUMP_1g516d6bb94a388c0efa9f50efa6d215c9>)

enumÂ [CUcoredumpSettings](<#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCoredumpGetAttribute](<#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd>) ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )
     Allows caller to fetch a coredump attribute value for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCoredumpGetAttributeGlobal](<#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819>) ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )
     Allows caller to fetch a coredump attribute value for the entire application.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCoredumpSetAttribute](<#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb>) ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )
     Allows caller to set a coredump attribute value for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCoredumpSetAttributeGlobal](<#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990>) ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )
     Allows caller to set a coredump attribute value globally.

### Enumerations

enum CUCoredumpGenerationFlags


Flags for controlling coredump contents

######  Values

CU_COREDUMP_DEFAULT_FLAGS = 0

CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = (1<<0)

CU_COREDUMP_SKIP_GLOBAL_MEMORY = (1<<1)

CU_COREDUMP_SKIP_SHARED_MEMORY = (1<<2)

CU_COREDUMP_SKIP_LOCAL_MEMORY = (1<<3)

CU_COREDUMP_SKIP_ABORT = (1<<4)

CU_COREDUMP_SKIP_CONSTBANK_MEMORY = (1<<5)

CU_COREDUMP_GZIP_COMPRESS = (1<<6)

CU_COREDUMP_LIGHTWEIGHT_FLAGS = CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES |CU_COREDUMP_SKIP_GLOBAL_MEMORY |CU_COREDUMP_SKIP_SHARED_MEMORY |CU_COREDUMP_SKIP_LOCAL_MEMORY |CU_COREDUMP_SKIP_CONSTBANK_MEMORY


enum CUcoredumpSettings


Flags for choosing a coredump attribute to get/set

######  Values

CU_COREDUMP_ENABLE_ON_EXCEPTION = 1

CU_COREDUMP_TRIGGER_HOST

CU_COREDUMP_LIGHTWEIGHT

CU_COREDUMP_ENABLE_USER_TRIGGER

CU_COREDUMP_FILE

CU_COREDUMP_PIPE

CU_COREDUMP_GENERATION_FLAGS

CU_COREDUMP_MAX


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCoredumpGetAttribute ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )


Allows caller to fetch a coredump attribute value for the current context.

######  Parameters

`attrib`
    \- The enum defining which value to fetch.
`value`
    \- void* containing the requested data.
`size`
    \- The size of the memory region `value` points to.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Returns in `*value` the requested value specified by `attrib`. It is up to the caller to ensure that the data type and size of `*value` matches the request.

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `attrib` will be placed in `size`.

The supported attributes are:

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false unless set to true globally or locally, or the CU_CTX_USER_COREDUMP_ENABLE flag was set during context creation.

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed.

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false unless set to true globally or locally. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead.

  * CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where true means that a coredump can be created by writing to the system pipe specified by CU_COREDUMP_PIPE. The default value is false unless set to true globally or locally.

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application.

  * CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe that will be monitored if user-triggered coredumps are enabled. The default value is corepipe.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA application and PID is the process ID of the CUDA application.

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior.


**See also:**

[cuCoredumpGetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819> "Allows caller to fetch a coredump attribute value for the entire application."), [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context."), [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCoredumpGetAttributeGlobal ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )


Allows caller to fetch a coredump attribute value for the entire application.

######  Parameters

`attrib`
    \- The enum defining which value to fetch.
`value`
    \- void* containing the requested data.
`size`
    \- The size of the memory region `value` points to.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*value` the requested value specified by `attrib`. It is up to the caller to ensure that the data type and size of `*value` matches the request.

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `attrib` will be placed in `size`.

The supported attributes are:

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false.

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed.

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead.

  * CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where true means that a coredump can be created by writing to the system pipe specified by CU_COREDUMP_PIPE. The default value is false.

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application.

  * CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe that will be monitored if user-triggered coredumps are enabled. The default value is corepipe.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA application and PID is the process ID of the CUDA application.

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior.


**See also:**

[cuCoredumpGetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd> "Allows caller to fetch a coredump attribute value for the current context."), [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context."), [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCoredumpSetAttribute ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )


Allows caller to set a coredump attribute value for the current context.

######  Parameters

`attrib`
    \- The enum defining which value to set.
`value`
    \- void* containing the requested data.
`size`
    \- The size of the memory region `value` points to.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

This function should be considered an alternate interface to the CUDA-GDB environment variables defined in this document: <https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump>

An important design decision to note is that any coredump environment variable values set before CUDA initializes will take permanent precedence over any values set with this function. This decision was made to ensure no change in behavior for any users that may be currently using these variables to get coredumps.

`*value` shall contain the requested value specified by `set`. It is up to the caller to ensure that the data type and size of `*value` matches the request.

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `set` will be placed in `size`.

/note This function will return [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>) if the caller attempts to set CU_COREDUMP_ENABLE_ON_EXCEPTION on a GPU of with Compute Capability < 6.0. [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") works on those platforms as an alternative.

/note CU_COREDUMP_ENABLE_USER_TRIGGER and CU_COREDUMP_PIPE cannot be set on a per-context basis.

The supported attributes are:

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false.

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed.

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead.

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application.

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior.


**See also:**

[cuCoredumpGetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819> "Allows caller to fetch a coredump attribute value for the entire application."), [cuCoredumpGetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd> "Allows caller to fetch a coredump attribute value for the current context."), [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCoredumpSetAttributeGlobal ( [CUcoredumpSettings](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g9b1cc417bdebfe4230e6dba3ea3d5b62>)Â attrib, void*Â value, size_t*Â size )


Allows caller to set a coredump attribute value globally.

######  Parameters

`attrib`
    \- The enum defining which value to set.
`value`
    \- void* containing the requested data.
`size`
    \- The size of the memory region `value` points to.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>)

###### Description

This function should be considered an alternate interface to the CUDA-GDB environment variables defined in this document: <https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump>

An important design decision to note is that any coredump environment variable values set before CUDA initializes will take permanent precedence over any values set with this function. This decision was made to ensure no change in behavior for any users that may be currently using these variables to get coredumps.

`*value` shall contain the requested value specified by `set`. It is up to the caller to ensure that the data type and size of `*value` matches the request.

If the caller calls this function with `*value` equal to NULL, the size of the memory region (in bytes) expected for `set` will be placed in `size`.

The supported attributes are:

  * CU_COREDUMP_ENABLE_ON_EXCEPTION: Bool where true means that GPU exceptions from this context will create a coredump at the location specified by CU_COREDUMP_FILE. The default value is false.

  * CU_COREDUMP_TRIGGER_HOST: Bool where true means that the host CPU will also create a coredump. The default value is true unless set to false globally or or locally. This value is deprecated as of CUDA 12.5 - raise the CU_COREDUMP_SKIP_ABORT flag to disable host device abort() if needed.

  * CU_COREDUMP_LIGHTWEIGHT: Bool where true means that any resulting coredumps will not have a dump of GPU memory or non-reloc ELF images. The default value is false. This attribute is deprecated as of CUDA 12.5, please use CU_COREDUMP_GENERATION_FLAGS instead.

  * CU_COREDUMP_ENABLE_USER_TRIGGER: Bool where true means that a coredump can be created by writing to the system pipe specified by CU_COREDUMP_PIPE. The default value is false.

  * CU_COREDUMP_FILE: String of up to 1023 characters that defines the location where any coredumps generated by this context will be written. The default value is core.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA applications and PID is the process ID of the CUDA application.

  * CU_COREDUMP_PIPE: String of up to 1023 characters that defines the name of the pipe that will be monitored if user-triggered coredumps are enabled. This value may not be changed after CU_COREDUMP_ENABLE_USER_TRIGGER is set to true. The default value is corepipe.cuda.HOSTNAME.PID where HOSTNAME is the host name of the machine running the CUDA application and PID is the process ID of the CUDA application.

  * CU_COREDUMP_GENERATION_FLAGS: An integer with values to allow granular control the data contained in a coredump specified as a bitwise OR combination of the following values: + CU_COREDUMP_DEFAULT_FLAGS - if set by itself, coredump generation returns to its default settings of including all memory regions that it is able to access + CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES \- Coredump will not include the data from CUDA source modules that are not relocated at runtime. + CU_COREDUMP_SKIP_GLOBAL_MEMORY \- Coredump will not include device-side global data that does not belong to any context. + CU_COREDUMP_SKIP_SHARED_MEMORY \- Coredump will not include grid-scale shared memory for the warp that the dumped kernel belonged to. + CU_COREDUMP_SKIP_LOCAL_MEMORY \- Coredump will not include local memory from the kernel. + CU_COREDUMP_LIGHTWEIGHT_FLAGS - Enables all of the above options. Equiavlent to setting the CU_COREDUMP_LIGHTWEIGHT attribute to true. + CU_COREDUMP_SKIP_ABORT - If set, GPU exceptions will not raise an abort() in the host CPU process. Same functional goal as CU_COREDUMP_TRIGGER_HOST but better reflects the default behavior.


**See also:**

[cuCoredumpGetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g56d7eb4975c7eb8e2b4eb0713fd8cedd> "Allows caller to fetch a coredump attribute value for the current context."), [cuCoredumpGetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g5cb5b7ddf41a2c3631eed8d00c4ae819> "Allows caller to fetch a coredump attribute value for the entire application."), [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context.")

* * *
