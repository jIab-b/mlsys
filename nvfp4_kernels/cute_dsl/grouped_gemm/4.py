import os
import re
import socket
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import cuda.bindings.runtime as cuda_runtime
import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl import detect_gpu_arch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    T,
    Uint32,
    dsl_user_op,
    extract_mlir_values,
    if_generate,
    min,
    new_from_mlir_values,
)
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    MbarrierArray,
    PipelineOp,
    PipelineState,
    PipelineTmaUmma,
    agent_sync,
    pipeline_init_arrive,
    pipeline_init_wait,
)

from task import input_t, output_t

mma_tile_instrs = 4
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
sf_container_dtype = cutlass.Int32
sf_elems_per_container = sf_container_dtype.width // sf_dtype.width
sf_rows = 128
SCHEDULER_DTYPE = Uint32
M_TILER_ENV = os.environ.get('M_TILER')
IS_2CTA = True
DEFAULT_CLUSTER_SHAPE = (2, 1)
DEFAULT_CLUSTER_M = DEFAULT_CLUSTER_SHAPE[0]
DEFAULT_CLUSTER_N = DEFAULT_CLUSTER_SHAPE[1]
CLUSTER_M = int(os.environ.get('CLUSTER_M', str(DEFAULT_CLUSTER_M)))
CLUSTER_N = int(os.environ.get('CLUSTER_N', str(DEFAULT_CLUSTER_N)))
DEFAULT_M_TILER = 256
M_TILER = 256
DEFAULT_N_TILER = 128
N_TILER = 128
DELEGATED_MBAR_INIT = os.environ.get('DELEGATED_MBAR_INIT', '0') == '1'
HOSTNAME = socket.gethostname()

@dsl_user_op
def cp_async_bulk_g2s(gmem_ptr: cute.Pointer, smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer, copy_bytes: cutlass.Int32, *, loc=None, ip=None):
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(None, [smem_ptr_i32, gmem_ptr_i64, cute.Int32(copy_bytes).ir_value(), mbar_ptr_i32], 'cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];', 'r,l,r,r', has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)

def encode_tcgen05_mma_descriptor_mxf4(kind: str='mxf4nvf4', m_dim: int=128, n_dim: int=8, k_dim: int=64, a_negate: bool=False, b_negate: bool=False, sparse: bool=False, scale_a_data_id: int=0, scale_b_data_id: int=0) -> int:
    if scale_a_data_id not in [0, 2]:
        raise ValueError(f'scale_a_data_id must be 0 or 2, got {scale_a_data_id}')
    if scale_b_data_id not in [0, 2]:
        raise ValueError(f'scale_b_data_id must be 0 or 2, got {scale_b_data_id}')
    if k_dim == 128 and (not sparse):
        raise ValueError('K=128 is only valid with sparse=True')
    if k_dim not in [64, 96, 128]:
        raise ValueError(f'k_dim must be 64, 96, or 128, got {k_dim}')
    if kind.lower() not in ['mxf4', 'mxf4nvf4']:
        raise ValueError(f"kind must be 'mxf4' or 'mxf4nvf4', got {kind}")
    descriptor = 0
    if sparse:
        descriptor |= 1 << 2
    descriptor |= (scale_b_data_id & 3) << 4
    descriptor |= (1 & 7) << 7
    descriptor |= (1 & 3) << 10
    if a_negate:
        descriptor |= 1 << 13
    if b_negate:
        descriptor |= 1 << 14
    n_shifted = n_dim >> 3 & 63
    descriptor |= n_shifted << 17
    if kind.lower() == 'mxf4':
        descriptor |= 1 << 23
    m_shifted = m_dim >> 7 & 3
    descriptor |= m_shifted << 27
    descriptor |= (scale_a_data_id & 3) << 29
    if k_dim == 96:
        descriptor |= 1 << 31
    return descriptor

def get_gpu_arch():
    if 'CUTE_DSL_ARCH' in os.environ:
        arch_str = os.environ['CUTE_DSL_ARCH']
        return arch_str
    return detect_gpu_arch('')

def get_kernel_base_path(kernel):
    dump_dir = Path(os.environ.get('CUTE_DSL_DUMP_DIR', '.'))
    function_name = kernel.function_name
    base = dump_dir / f'{function_name}.{get_gpu_arch()}'
    return base

@dsl_user_op
def vote_ballot_u32(predicate: Boolean, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(T.i32(), [Boolean(predicate).ir_value(loc=loc, ip=ip)], '{\n            .reg .pred p;\n            setp.ne.b32 p, $1, 0;\n            vote.sync.ballot.b32 $0, p, 0xFFFFFFFF;\n            }', '=r,r', has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))

def optimize_ptx_smem_base(ptx_content: str, max_mbarrier_distance: int=5, verbose: bool=False) -> tuple[str, str | None]:
    lines = ptx_content.split('\n')
    reg_decl_pattern = re.compile('(\\.reg\\s+\\.b32\\s+%r<)(\\d+)(>;)')
    smem_pattern = re.compile('mov\\.u32\\s+(%r\\d+),\\s+__dynamic_shmem__0;')
    tid_pattern = re.compile('mov\\.u32\\s+%r\\d+,\\s+%tid\\.x;')
    reg_decl_line = None
    new_smem_reg = None
    for i, line in enumerate(lines):
        match = reg_decl_pattern.search(line)
        if match:
            reg_decl_line = i
            old_count = int(match.group(2))
            new_count = old_count + 1
            new_smem_reg = f'%r{old_count}'
            lines[i] = reg_decl_pattern.sub(f'\\g<1>{new_count}\\g<3>', line)
            break
    if reg_decl_line is None or new_smem_reg is None:
        if verbose:
            print('  smem optimization: could not find register declaration')
        return (ptx_content, None)
    inject_line = None
    for i, line in enumerate(lines):
        if tid_pattern.search(line):
            inject_line = i + 1
            break
    if inject_line is None:
        if verbose:
            print('  smem optimization: could not find injection point')
        return (ptx_content, None)
    indent = '\t' * 2
    smem_load_instruction = f'{indent}mov.u32 \t{new_smem_reg}, __dynamic_shmem__0;'
    lines.insert(inject_line, smem_load_instruction)
    count = 0
    for i in range(inject_line + 1, len(lines)):
        match = smem_pattern.search(lines[i])
        if match:
            reg = match.group(1)
            lines[i] = smem_pattern.sub(f'mov.u32 \t{reg}, {new_smem_reg};', lines[i])
            count += 1
    if verbose:
        print(f'  smem optimization: injected {new_smem_reg}, replaced {count} __dynamic_shmem__0 loads')
    return ('\n'.join(lines), new_smem_reg)

@dsl_user_op
def popc_u32(value: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(T.i32(), [Uint32(value).ir_value(loc=loc, ip=ip)], 'popc.b32 $0, $1;', '=r,r', has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))

def make_warp_uniform_u32(value: Uint32, *, loc=None, ip=None) -> Uint32:
    import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
    return Uint32(_cute_nvgpu_ir.arch_make_warp_uniform(Uint32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))

class PTXOptimizer:
    _instance: 'PTXOptimizer | None' = None
    _original_load_cuda_library = None
    _enabled: bool = False
    _verbose: bool = False
    _apply_smem_opt: bool = True
    _ptx_version: str = '9.1'
    _opt_level: str = 'O3'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _get_ptx_for_compiled_func(cls, compiled_func) -> tuple[str, Path] | None:
        import time
        func_name = getattr(compiled_func, 'function_name', None)
        if not func_name:
            return None
        dump_dir = Path(os.environ.get('CUTE_DSL_DUMP_DIR', Path.cwd()))
        patterns = [f'{func_name}.sm_*.ptx', f'*{func_name}*.ptx']
        max_retries = 10
        for retry in range(max_retries):
            for pattern in patterns:
                matches = list(dump_dir.glob(pattern))
                for ptx_path in matches:
                    if '.optimized' in ptx_path.name:
                        continue
                    content = ptx_path.read_text().rstrip('\x00')
                    has_entry = '.entry ' in content
                    has_closing_brace = '}\n' in content or content.rstrip().endswith('}')
                    if has_entry and has_closing_brace:
                        return (content, ptx_path)
            if retry < max_retries - 1:
                time.sleep(0.1)
        if cls._verbose:
            print(f'[ptx-opt] No valid PTX found in {dump_dir}')
        return None

    @classmethod
    def _compile_ptx_with_optimization(cls, ptx_path: Path, ptx_content: str) -> bytes:
        if cls._apply_smem_opt:
            optimized_ptx, smem_reg = optimize_ptx_smem_base(ptx_content, verbose=cls._verbose)
        else:
            optimized_ptx = ptx_content
            if cls._verbose:
                print('[ptx-opt] Skipping smem optimization (disabled)')
        match = re.search('\\.target\\s+(sm_\\d+[a-z]?)', optimized_ptx)
        arch = match.group(1) if match else 'sm_110a'
        arch_mapping = {'sm_101': 'sm_110', 'sm_101a': 'sm_110a', 'sm_101f': 'sm_110f'}
        original_arch = arch
        arch = arch_mapping.get(arch, arch)
        if original_arch in arch_mapping:
            optimized_ptx = re.sub('\\.version\\s+\\d+\\.\\d+', f'.version {cls._ptx_version}', optimized_ptx)
            optimized_ptx = re.sub(f'\\.target\\s+{re.escape(original_arch)}', f'.target {arch}', optimized_ptx)
        optimized_ptx = re.sub('^\\s*\\.loc\\s+\\d+\\s+\\d+\\s+\\d+\\s*\\n', '', optimized_ptx, flags=re.MULTILINE)
        optimized_ptx = re.sub('^\\s*\\.file\\s+.*\\n', '', optimized_ptx, flags=re.MULTILINE)
        optimized_ptx_path = ptx_path.with_suffix('.optimized.ptx')
        optimized_ptx_path.write_text(optimized_ptx)
        cubin_path = ptx_path.with_suffix('.optimized.cubin')
        try:
            result = subprocess.run(['ptxas', f'-arch={arch}', f'-{cls._opt_level}', '-o', str(cubin_path), str(optimized_ptx_path)], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'ptxas failed: {result.stderr}')
            cubin_data = cubin_path.read_bytes()
            return cubin_data
        except FileNotFoundError:
            raise RuntimeError('ptxas not found in PATH')

    @classmethod
    def _make_patched_load_cuda_library(cls):

        def _patched_load_cuda_library(self):
            import ctypes
            result = cls._get_ptx_for_compiled_func(self)
            if not result:
                if cls._verbose:
                    print('[ptx-opt] PTX not found, using embedded cubin')
                return cls._original_load_cuda_library(self)
            ptx_content, ptx_path = result
            try:
                cubin = cls._compile_ptx_with_optimization(ptx_path, ptx_content)
            except Exception as e:
                if cls._verbose:
                    print(f'[ptx-opt] Optimization failed ({e}), using embedded cubin')
                return cls._original_load_cuda_library(self)
            err, library = cuda_runtime.cudaLibraryLoadData(cubin, None, None, 0, None, None, 0)
            if err != cuda_runtime.cudaError_t.cudaSuccess:
                if cls._verbose:
                    print(f'[ptx-opt] cudaLibraryLoadData failed ({err}), using embedded cubin')
                return cls._original_load_cuda_library(self)
            _, cuda_load_to_device = self._get_cuda_init_and_load()
            lib_ptr = ctypes.c_void_p(int(library))
            dev_id = ctypes.c_int32(0)
            err_val = ctypes.c_int32(0)
            args = (ctypes.c_void_p * 3)(ctypes.cast(ctypes.pointer(lib_ptr), ctypes.c_void_p), ctypes.cast(ctypes.pointer(dev_id), ctypes.c_void_p), ctypes.cast(ctypes.pointer(err_val), ctypes.c_void_p))
            for dev in range(self.num_devices):
                dev_id.value = dev
                cuda_load_to_device(args)
                if err_val.value != 0:
                    if cls._verbose:
                        print(f'[ptx-opt] cuda_load_to_device failed on device {dev}, using embedded cubin')
                    return cls._original_load_cuda_library(self)
            return [cuda_runtime.cudaLibrary_t(lib_ptr.value)]
        return _patched_load_cuda_library

    @classmethod
    def enable(cls, verbose: bool=False, apply_smem_opt: bool=True, ptx_version: str='9.1', opt_level: str='O3'):
        if cls._enabled:
            return
        if os.environ.get('CUTE_DSL_KEEP_PTX', '0') != '1':
            os.environ['CUTE_DSL_KEEP_PTX'] = '1'
            if verbose:
                print('[ptx-opt] Set CUTE_DSL_KEEP_PTX=1')
        cls._verbose = verbose
        cls._apply_smem_opt = apply_smem_opt
        cls._ptx_version = ptx_version
        cls._opt_level = opt_level
        from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
        cls._original_load_cuda_library = CudaDialectJitCompiledFunction._load_cuda_library
        CudaDialectJitCompiledFunction._load_cuda_library = cls._make_patched_load_cuda_library()
        cls._enabled = True
        if verbose:
            print('[ptx-opt] PTX optimization patch enabled')

    @classmethod
    def disable(cls):
        if not cls._enabled:
            return
        from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
        CudaDialectJitCompiledFunction._load_cuda_library = cls._original_load_cuda_library
        cls._enabled = False
        if cls._verbose:
            print('[ptx-opt] PTX optimization patch disabled')

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

@dsl_user_op
def clusterlaunchcontrol_try_cancel(response_smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer, *, multicast: bool=False, loc=None, ip=None) -> None:
    response_ptr_i32 = response_smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    multicast_qualifier = '.multicast::cluster::all' if multicast else ''
    llvm.inline_asm(None, [response_ptr_i32, mbar_ptr_i32], f'clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{multicast_qualifier}.b128 [$0], [$1];', 'r,r', has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)

def cluster_shape_to_bulk_atom_A(cluster_shape_mnk: cute.Shape, atom_thr_id: cute.Layout) -> Union[cpasync.CopyBulkG2SMulticastOp, cpasync.CopyBulkG2SOp]:
    atom_sm_cnt = cute.size(atom_thr_id)
    mcast = not cute.size(cluster_shape_mnk, mode=[1]) == 1
    cluster_size = cute.size(cluster_shape_mnk)
    if not isinstance(cluster_size, int) or not isinstance(atom_sm_cnt, int):
        raise ValueError(f'Dynamic cluster shape or atom SM count is not supported: {cluster_shape_mnk} and {atom_thr_id}')
    if cute.size(cluster_shape_mnk, mode=[0]) % atom_sm_cnt != 0:
        raise ValueError(f'Cluster shape not divisible by MMA size: {cluster_shape_mnk} and {atom_thr_id}')
    if mcast:
        return cpasync.CopyBulkG2SMulticastOp()
    else:
        return cpasync.CopyBulkG2SOp()

@dsl_user_op
def clusterlaunchcontrol_query_cancel(response_smem_ptr: cute.Pointer, *, loc=None, ip=None) -> tuple[Int32, Int32, Int32, cute.Boolean]:
    response_ptr_i32 = response_smem_ptr.toint(loc=loc, ip=ip).ir_value()
    result_type = ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>')
    results = llvm.inline_asm(result_type, [response_ptr_i32], '{\n.reg .pred p1;\n\t.reg .b128 clc_result;\n\tld.shared.b128 clc_result, [$4];\n\tclusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\tselp.u32 $3, 1, 0, p1;\n\t@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;\n\t}\n', '=r,=r,=r,=r,r', has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    cute.arch.fence_view_async_shared()
    ctaid_x = Int32(llvm.extractvalue(T.i32(), results, [0], loc=loc, ip=ip))
    ctaid_y = Int32(llvm.extractvalue(T.i32(), results, [1], loc=loc, ip=ip))
    ctaid_z = Int32(llvm.extractvalue(T.i32(), results, [2], loc=loc, ip=ip))
    is_valid = cute.Boolean(llvm.extractvalue(T.i32(), results, [3], loc=loc, ip=ip))
    return (ctaid_x, ctaid_y, ctaid_z, is_valid)

def cluster_shape_to_bulk_atom_SFB(cluster_shape_mnk: cute.Shape, atom_thr_id: cute.Layout) -> Union[cpasync.CopyBulkG2SMulticastOp, cpasync.CopyBulkG2SOp]:
    atom_sm_cnt = cute.size(atom_thr_id)
    mcast = not cute.size(cluster_shape_mnk, mode=[0]) == 1
    cluster_size = cute.size(cluster_shape_mnk)
    if not isinstance(cluster_size, int) or not isinstance(atom_sm_cnt, int):
        raise ValueError(f'Dynamic cluster shape or atom SM count is not supported: {cluster_shape_mnk} and {atom_thr_id}')
    if cute.size(cluster_shape_mnk, mode=[0]) % atom_sm_cnt != 0:
        raise ValueError(f'Cluster shape not divisible by MMA size: {cluster_shape_mnk} and {atom_thr_id}')
    if atom_sm_cnt == 2:
        return cpasync.CopyBulkG2SMulticastOp()
    elif atom_sm_cnt == 1 and mcast:
        return cpasync.CopyBulkG2SMulticastOp()
    elif atom_sm_cnt == 1 and (not mcast):
        return cpasync.CopyBulkG2SOp()
    raise ValueError(f'Unsupported Configuration for bulk copy: {cluster_shape_mnk} and {atom_thr_id}')

def enable_ptx_optimizer(verbose: bool=False, apply_smem_opt: bool=True, ptx_version: str='9.1', opt_level: str='O3'):
    PTXOptimizer.enable(verbose=verbose, apply_smem_opt=apply_smem_opt, ptx_version=ptx_version, opt_level=opt_level)

def cute_compile(*args, rename: Optional[str]=None, keep_cubin: bool=True, keep_ptx: bool=True, gen_line_info: bool=True, strip_ptx_line_info: bool=True, latest_ptxas: bool=True, options: Optional[str]=None, **kwargs):
    option_strings = []
    if keep_cubin:
        option_strings.append('--keep-cubin')
    if keep_ptx:
        option_strings.append('--keep-ptx')
    if gen_line_info:
        option_strings.append('--generate-line-info')
    if options is not None:
        option_strings.append(options)
    try:
        kernel = cute.compile(*args, options=' '.join(option_strings) if option_strings else None, **kwargs)
    except TypeError:
        kernel = cute.compile(*args, **kwargs)
    if HOSTNAME == 'thor':
        base = get_kernel_base_path(kernel)
        original_ptx_path = Path(f'{base}.ptx')
        cubin_path = Path(f'{base}.cubin')
        ptx_path = Path(f'{base}.ptx')
        if rename:
            import shutil
            new_cubin_path = cubin_path.parent / f'{rename}.cubin'
            new_ptx_path = ptx_path.parent / f'{rename}.ptx'
            if cubin_path.exists():
                shutil.copy2(cubin_path, new_cubin_path)
            if ptx_path.exists():
                shutil.copy2(ptx_path, new_ptx_path)
            cubin_path = new_cubin_path
            ptx_path = new_ptx_path
        import multiprocessing

        def run_latest_ptxas(ptx_path, cubin_path, ptx_version, opt_level):
            import re
            import subprocess

            from cute_utils.compile import optimize_ptx_smem_base
            with open(ptx_path, 'r') as f:
                ptx_content = f.read()
            ptx_content = ptx_content.replace('\x00', '')
            ptx_content = re.sub('^\\s*\\.loc\\s+\\d+\\s+\\d+\\s+\\d+\\s*\\n', '', ptx_content, flags=re.MULTILINE)
            ptx_content = re.sub('^\\s*\\.file\\s+.*\\n', '', ptx_content, flags=re.MULTILINE)
            target_match = re.search('^\\.target\\s+(sm_\\w+)', ptx_content, flags=re.MULTILINE)
            original_arch = target_match.group(1) if target_match else 'sm_100a'
            arch_mapping = {'sm_101': 'sm_110', 'sm_101a': 'sm_110a', 'sm_101f': 'sm_110f'}
            target_arch = arch_mapping.get(original_arch, original_arch)
            ptx_content = re.sub('^\\.version\\s+\\d+\\.\\d+', f'.version {ptx_version}', ptx_content, flags=re.MULTILINE)
            if original_arch != target_arch:
                ptx_content = re.sub(f'^\\.target\\s+{re.escape(original_arch)}', f'.target {target_arch}', ptx_content, flags=re.MULTILINE)
            out_dir = ptx_path.parent / f'ptx_{ptx_version}'
            out_dir.mkdir(exist_ok=True)
            ptx_ver_path = out_dir / f'{ptx_path.stem}_{ptx_version}.ptx'
            with open(ptx_ver_path, 'w') as f:
                f.write(ptx_content)
            cubin_ver_path = out_dir / f'{cubin_path.stem}_{ptx_version}.cubin'
            try:
                subprocess.run(['ptxas', f'-arch={target_arch}', f'-{opt_level}', str(ptx_ver_path), '-o', str(cubin_ver_path)], capture_output=True, text=True, check=True)
                asm_ver_path = cubin_ver_path.with_suffix('.asm')
                result = subprocess.run(['/home/sonny/blog_scripts/sass/disasm', str(cubin_ver_path)], capture_output=True, text=True, check=True)
                with open(asm_ver_path, 'w') as f:
                    f.write(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f'Error with latest ptxas compilation: {e}')
                if e.stderr:
                    print(f'stderr: {e.stderr}')
            optimized_ptx_content, smem_base_reg = optimize_ptx_smem_base(ptx_content)
            if smem_base_reg is not None:
                optimized_ptx_path = ptx_path.with_suffix('.optimized.ptx')
                with open(optimized_ptx_path, 'w') as f:
                    f.write(optimized_ptx_content)
                optimized_cubin_path = ptx_path.with_suffix('.optimized.cubin')
                try:
                    subprocess.run(['ptxas', f'-arch={target_arch}', f'-{opt_level}', str(optimized_ptx_path), '-o', str(optimized_cubin_path)], capture_output=True, text=True, check=True)
                    optimized_asm_path = ptx_path.with_suffix('.optimized.asm')
                    result = subprocess.run(['/home/sonny/blog_scripts/sass/disasm', str(optimized_cubin_path)], capture_output=True, text=True, check=True)
                    with open(optimized_asm_path, 'w') as f:
                        f.write(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f'Error compiling optimized PTX: {e}')
                    if e.stderr:
                        print(f'stderr: {e.stderr}')

        def run_disasm_and_strip(cubin_path, ptx_path, gen_line_info, strip_ptx_line_info):
            import re
            import subprocess
            asm_path = cubin_path.with_suffix('.asm')
            if cubin_path.exists():
                try:
                    result = subprocess.run(['/home/sonny/blog_scripts/sass/disasm', str(cubin_path)], capture_output=True, text=True, check=True)
                    with open(asm_path, 'w') as f:
                        f.write(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f'Error disassembling cubin: {e}')
            if gen_line_info and strip_ptx_line_info and ptx_path.exists():
                loc_pattern = re.compile('^\\s*\\.loc\\s+\\d+\\s+\\d+\\s+\\d+\\s*$')
                file_pattern = re.compile('^\\s*\\.file\\s+.*$')
                with open(ptx_path, 'r') as f:
                    lines = f.readlines()
                filtered_lines = [line for line in lines if not loc_pattern.match(line) and (not file_pattern.match(line))]
                with open(ptx_path, 'w') as f:
                    f.writelines(filtered_lines)

        def run_copy_optimized_files(original_base_path, renamed_ptx_path):
            import shutil
            import subprocess
            import time
            optimized_ptx_src = original_base_path.with_suffix('.optimized.ptx')
            optimized_cubin_src = original_base_path.with_suffix('.optimized.cubin')
            max_wait = 10
            for _ in range(max_wait * 10):
                if optimized_ptx_src.exists() and optimized_cubin_src.exists():
                    break
                time.sleep(0.1)
            else:
                return
            renamed_base = renamed_ptx_path.with_suffix('')
            optimized_ptx_dst = Path(str(renamed_base) + '.optimized.ptx')
            optimized_cubin_dst = Path(str(renamed_base) + '.optimized.cubin')
            optimized_asm_dst = Path(str(renamed_base) + '.optimized.asm')
            shutil.copy2(optimized_ptx_src, optimized_ptx_dst)
            shutil.copy2(optimized_cubin_src, optimized_cubin_dst)
            try:
                result = subprocess.run(['/home/sonny/blog_scripts/sass/disasm', str(optimized_cubin_dst)], capture_output=True, text=True, check=True)
                with open(optimized_asm_dst, 'w') as f:
                    f.write(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f'Error disassembling optimized cubin: {e}')
        current_process = multiprocessing.current_process()
        is_daemon = getattr(current_process, 'daemon', False)
        if not is_daemon:
            if latest_ptxas and ptx_path.exists():
                p1 = multiprocessing.Process(target=run_latest_ptxas, args=(ptx_path, cubin_path, PTXOptimizer._ptx_version, PTXOptimizer._opt_level))
                p1.start()
            p2 = multiprocessing.Process(target=run_disasm_and_strip, args=(cubin_path, ptx_path, gen_line_info, strip_ptx_line_info))
            p2.start()
            if rename and original_ptx_path is not None:
                p3 = multiprocessing.Process(target=run_copy_optimized_files, args=(original_ptx_path, ptx_path))
                p3.start()
    return kernel

def cute_compile_optimized(*args, verbose: bool=False, apply_smem_opt: bool=False, ptx_version: str='9.1', opt_level: str='O3', **kwargs):
    enable_ptx_optimizer(verbose=verbose, apply_smem_opt=apply_smem_opt, ptx_version=ptx_version, opt_level=opt_level)
    return cute_compile(*args, **kwargs)

@dsl_user_op
def tcgen05_mma_1cta_fp4(acc_tmem_ptr: cute.Pointer, A_desc: cute.Int64, B_desc: cute.Int64, I_desc: cute.Int32, SFA_tmem_ptr: cute.Pointer, SFB_tmem_ptr: cute.Pointer, pred: cute.Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [acc_tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), A_desc.ir_value(loc=loc, ip=ip), B_desc.ir_value(loc=loc, ip=ip), cute.Int32(I_desc).ir_value(loc=loc, ip=ip), SFA_tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), SFB_tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), cute.Int32(pred).ir_value(loc=loc, ip=ip)], '{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [$0], $1, $2, $3, [$4], [$5], p;\n\t}\n', 'r,l,l,r,r,r,r', has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)

@dsl_user_op
def tcgen05_mma_2cta_fp4(acc_tmem_ptr: cute.Pointer, A_desc: cute.Int64, B_desc: cute.Int64, I_desc: cute.Int32, SFA_tmem_ptr: cute.Pointer, SFB_tmem_ptr: cute.Pointer, pred: cute.Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [acc_tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), A_desc.ir_value(loc=loc, ip=ip), B_desc.ir_value(loc=loc, ip=ip), cute.Int32(I_desc).ir_value(loc=loc, ip=ip), SFA_tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), SFB_tmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), cute.Int32(pred).ir_value(loc=loc, ip=ip)], '{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [$0], $1, $2, $3, [$4], [$5], p;\n\t}\n', 'r,l,l,r,r,r,r', has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)

def smem_desc_encode(x):
    return (x & 262143) >> 4

@cute.jit
def tcgen05_encode_smem_addr_in_desc(base_desc: cute.Int64, addr: cute.Pointer):
    addr_encoded = smem_desc_encode(addr.toint().to(cute.Int64))
    desc = base_desc + addr_encoded
    return desc

@cute.jit
def tcgen05_encode_base_smem_desc(ld=16, stride=0, matrix_base_offset=0, swizzle=0, ld_abs=False) -> cute.Int64:
    desc = smem_desc_encode(ld) << 16 | smem_desc_encode(stride) << 32 | 1 << 46 | matrix_base_offset << 49 | (1 if ld_abs else 0) << 52 | 11 << 53 | swizzle << 61
    return desc

def is_power_of_2(x: int) -> bool:
    return x > 0 and x & x - 1 == 0

@dsl_user_op
def make_smem_layout_sfa(tiled_mma: cute.TiledMma, mma_tiler_mnk: cute.Tile, sf_vec_size: int, num_stages: int, *, loc=None, ip=None) -> cute.Layout:
    sfa_tile_shape = (128, mma_tiler_mnk[2])
    smem_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(sf_vec_size).layout, sfa_tile_shape, (2, 1))
    mma_tile_inst_k = 4
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)
    atom_m = 128
    tiler_inst = ((atom_m, sf_vec_size),)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)
    sfa_smem_layout_staged = cute.append(smem_layout, cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))))
    return sfa_smem_layout_staged

@dsl_user_op
def make_tmem_layout_sfb(tiled_mma: cute.TiledMma, mma_tiler_mnk: cute.Tile, sf_vec_size: int, smem_layout: Union[cute.Layout, cute.ComposedLayout], *, loc=None, ip=None) -> cute.Layout:
    atom_thr_size = cute.size(tiled_mma.thr_id.shape, loc=loc, ip=ip)
    cta_tile_shape_m = mma_tiler_mnk[0] // atom_thr_size
    sfb_layout_ty = _cute_nvgpu_ir.make_tmem_layout_sfb(smem_layout, cta_tile_shape_m, atom_thr_size, sf_vec_size)
    return _cute_ir.static(sfb_layout_ty, loc=loc, ip=ip)
DTYPE = SCHEDULER_DTYPE

class LookupMode(IntEnum):
    SINGLE_WITH_TRAILING = 1
    TWO_MODE = 2
    POPCOUNT = 3

@dataclass(frozen=True)
class TileSchedulerParams:
    cluster_prefix_sum: Tuple[int, ...]
    cluster_counts_m: Tuple[int, ...]
    cluster_counts_n: Tuple[int, ...]
    total_clusters: int
    num_active_clusters: int
    cluster_shape_mn: Tuple[int, int]
    raster_along_m: bool
    num_groups: int
    lookup_mode: int
    uniform_tiles_per_group: int
    uniform_group_count: int
    uniform_total_tiles: int
    mode2_tiles_per_group: int
    first_iter_can_use_divide: bool
    use_clc: bool = False
    n_tiles_per_group: Tuple[int, ...] = ()
    split_residual_swap_per_group: Tuple[bool, ...] = ()

    @property
    def lookup_mode_name(self) -> str:
        return {LookupMode.SINGLE_WITH_TRAILING: 'SINGLE_WITH_TRAILING', LookupMode.TWO_MODE: 'TWO_MODE', LookupMode.POPCOUNT: 'POPCOUNT'}.get(self.lookup_mode, f'UNKNOWN({self.lookup_mode})')

    def __str__(self) -> str:
        lines = ['TileSchedulerParams:', f'  num_groups: {self.num_groups}', f'  total_clusters: {self.total_clusters}', f'  num_active_clusters: {self.num_active_clusters}', f'  cluster_shape_mn: {self.cluster_shape_mn}', f'  raster_along_m: {self.raster_along_m}', f'  cluster_prefix_sum: {self.cluster_prefix_sum}', f'  cluster_counts_m: {self.cluster_counts_m}', f'  cluster_counts_n: {self.cluster_counts_n}', f'  lookup_mode: {self.lookup_mode_name}', f'  uniform_tiles_per_group: {self.uniform_tiles_per_group}', f'  uniform_group_count: {self.uniform_group_count}', f'  uniform_total_tiles: {self.uniform_total_tiles}', f'  mode2_tiles_per_group: {self.mode2_tiles_per_group}', f'  first_iter_can_use_divide: {self.first_iter_can_use_divide}', f'  use_clc: {self.use_clc}', f'  n_tiles_per_group: {self.n_tiles_per_group}', f'  split_residual_swap_per_group: {self.split_residual_swap_per_group}']
        return '\n'.join(lines)

    @staticmethod
    def compute_lookup_mode(cluster_prefix_sum: Tuple[int, ...], num_groups: int, num_active_clusters: int) -> Tuple[int, int, int, int, int, bool]:
        if num_groups == 0:
            return (LookupMode.SINGLE_WITH_TRAILING, 0, 0, 0, 0, True)
        tile_counts = []
        prev = 0
        for psum in cluster_prefix_sum:
            tile_counts.append(psum - prev)
            prev = psum
        first_count = tile_counts[0]
        uniform_group_count = 1
        for i in range(1, num_groups):
            if tile_counts[i] == first_count:
                uniform_group_count += 1
            else:
                break
        uniform_tiles_per_group = first_count
        uniform_total_tiles = uniform_group_count * uniform_tiles_per_group
        if uniform_group_count >= num_groups - 1:
            return (LookupMode.SINGLE_WITH_TRAILING, uniform_tiles_per_group, uniform_group_count, uniform_total_tiles, 0, True)
        second_count = tile_counts[uniform_group_count]
        all_remaining_same = all((tile_counts[i] == second_count for i in range(uniform_group_count, num_groups)))
        first_iter_can_use_divide = num_active_clusters <= uniform_total_tiles
        if all_remaining_same:
            return (LookupMode.TWO_MODE, uniform_tiles_per_group, uniform_group_count, uniform_total_tiles, second_count, first_iter_can_use_divide)
        else:
            return (LookupMode.POPCOUNT, uniform_tiles_per_group, uniform_group_count, uniform_total_tiles, 0, first_iter_can_use_divide)

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

class SimpleWorkTileInfo:

    def __init__(self, group_idx: Uint32, cta_tile_m: Uint32, cta_tile_n: Uint32, is_valid: Boolean):
        self._group_idx = group_idx
        self._cta_tile_m = cta_tile_m
        self._cta_tile_n = cta_tile_n
        self._is_valid = Boolean(is_valid)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self._group_idx)
        values.extend(extract_mlir_values(self._cta_tile_m))
        values.extend(extract_mlir_values(self._cta_tile_n))
        values.extend(extract_mlir_values(self._is_valid))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> 'SimpleWorkTileInfo':
        assert len(values) == 4
        new_group_idx = new_from_mlir_values(self._group_idx, [values[0]])
        new_cta_tile_m = new_from_mlir_values(self._cta_tile_m, [values[1]])
        new_cta_tile_n = new_from_mlir_values(self._cta_tile_n, [values[2]])
        new_is_valid = new_from_mlir_values(self._is_valid, [values[3]])
        return SimpleWorkTileInfo(new_group_idx, new_cta_tile_m, new_cta_tile_n, new_is_valid)

    @property
    def group_idx(self) -> Uint32:
        return self._group_idx

    @property
    def cta_tile_m(self) -> Uint32:
        return self._cta_tile_m

    @property
    def cta_tile_n(self) -> Uint32:
        return self._cta_tile_n

    @property
    def is_valid(self) -> Boolean:
        return self._is_valid

    @property
    def is_valid_tile(self) -> Boolean:
        return self._is_valid

    @property
    def cta_tile_idx_m(self) -> Uint32:
        return self._cta_tile_m

    @property
    def cta_tile_idx_n(self) -> Uint32:
        return self._cta_tile_n

@dataclass(frozen=True)
class PipelineClcFetchAsync:
    sync_object_full: MbarrierArray
    sync_object_empty: MbarrierArray
    num_stages: int
    producer_mask: Optional[Int32]
    consumer_mask: Optional[Int32]
    is_signalling_thread: Boolean

    @staticmethod
    @cute.jit
    def _init_full_barrier_arrive_signal(cta_layout_vmnk: cute.Layout, tidx: Int32):
        dst_rank = tidx % 32
        is_signalling_thread = dst_rank < cute.size(cta_layout_vmnk)
        return (dst_rank, is_signalling_thread)

    @staticmethod
    def create(*, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, tx_count: int, barrier_storage: cute.Pointer=None, producer_mask: Int32=None, consumer_mask: Int32=None, cta_layout_vmnk: Optional[cute.Layout]=None, defer_sync: bool=False):
        if not isinstance(barrier_storage, cute.Pointer):
            raise TypeError(f'Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}')
        fake_producer_type = PipelineOp.AsyncThread
        consumer_type = PipelineOp.AsyncThread
        producer = (fake_producer_type, producer_group)
        consumer = (consumer_type, consumer_group)
        sync_object_full = PipelineTmaUmma._make_sync_object(barrier_storage.align(min_align=8), num_stages, producer, tx_count)
        sync_object_empty = PipelineTmaUmma._make_sync_object(barrier_storage.align(min_align=8) + num_stages, num_stages, consumer)
        if cta_layout_vmnk is None:
            cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tidx, _, _ = cute.arch.thread_idx()
        producer_mask, is_signalling_thread = PipelineClcFetchAsync._init_full_barrier_arrive_signal(cta_layout_vmnk, tidx)
        consumer_mask = 0
        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)
        return PipelineClcFetchAsync(sync_object_full, sync_object_empty, num_stages, producer_mask, consumer_mask, is_signalling_thread)

    @dsl_user_op
    def producer_acquire(self, state: PipelineState, try_acquire_token: Optional[Boolean]=None, *, loc=None, ip=None):
        if_generate(try_acquire_token is None or try_acquire_token == 0, lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip), loc=loc, ip=ip)

        def _arrive_clc_full():
            barrier_ptr = self.sync_object_full.get_barrier(state.index, loc=loc, ip=ip)
            tx_count = self.sync_object_full.tx_count
            cute.arch.mbarrier_arrive_and_expect_tx(barrier_ptr, tx_count, self.producer_mask, loc=loc, ip=ip)
        if_generate(self.is_signalling_thread, _arrive_clc_full, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait(self, state: PipelineState, try_wait_token: Optional[Boolean]=None, *, loc=None, ip=None):
        if_generate(try_wait_token is None or try_wait_token == 0, lambda: self.sync_object_full.wait(state.index, state.phase, loc=loc, ip=ip), loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        self.sync_object_empty.arrive(state.index, self.consumer_mask, loc=loc, ip=ip)

    @dsl_user_op
    def producer_get_barrier(self, state: PipelineState, *, loc=None, ip=None) -> cute.Pointer:
        return self.sync_object_full.get_barrier(state.index, loc=loc, ip=ip)

    @dsl_user_op
    def producer_tail(self, state: PipelineState, try_acquire_token: Optional[Boolean]=None, *, loc=None, ip=None):
        for i in range(self.num_stages):
            if_generate(try_acquire_token is None or try_acquire_token == 0, lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip), loc=loc, ip=ip)
            state.advance(loc=loc, ip=ip)

class SimpleGroupedTileScheduler:

    def __init__(self, params: TileSchedulerParams, group_agnostic: cutlass.Constexpr[bool], linear_idx: Uint32, cta_in_cluster_m: Uint32, cta_in_cluster_n: Uint32, num_tiles_executed: Uint32, lane_idx: Uint32, my_psum: Uint32, my_cluster_count_m: Uint32, my_cluster_count_n: Uint32, clc_response_ptr: Optional[cute.Pointer]=None):
        self.params = params
        self.group_agnostic = group_agnostic
        self._linear_idx = make_warp_uniform_u32(linear_idx)
        self._cta_in_cluster_m = make_warp_uniform_u32(cta_in_cluster_m)
        self._cta_in_cluster_n = make_warp_uniform_u32(cta_in_cluster_n)
        self._num_tiles_executed = make_warp_uniform_u32(num_tiles_executed)
        self._lane_idx = lane_idx
        self._my_psum = my_psum
        self._my_cluster_count_m = my_cluster_count_m
        self._my_cluster_count_n = my_cluster_count_n
        self._clc_response_ptr = clc_response_ptr

    @property
    def use_clc(self):
        return self.params.use_clc

    @property
    def total_clusters(self):
        return self.params.total_clusters

    @property
    def num_active_clusters(self):
        return self.params.num_active_clusters

    @property
    def cluster_shape_mn(self):
        return self.params.cluster_shape_mn

    @property
    def raster_along_m(self):
        return self.params.raster_along_m

    @property
    def num_groups(self):
        return self.params.num_groups

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self._linear_idx)
        values.extend(extract_mlir_values(self._cta_in_cluster_m))
        values.extend(extract_mlir_values(self._cta_in_cluster_n))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self._lane_idx))
        values.extend(extract_mlir_values(self._my_psum))
        values.extend(extract_mlir_values(self._my_cluster_count_m))
        values.extend(extract_mlir_values(self._my_cluster_count_n))
        if cutlass.const_expr(self.use_clc):
            values.extend(extract_mlir_values(self._clc_response_ptr))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> 'SimpleGroupedTileScheduler':
        expected_len = 9 if self.use_clc else 8
        assert len(values) == expected_len
        new_linear_idx = new_from_mlir_values(self._linear_idx, [values[0]])
        new_cta_in_cluster_m = new_from_mlir_values(self._cta_in_cluster_m, [values[1]])
        new_cta_in_cluster_n = new_from_mlir_values(self._cta_in_cluster_n, [values[2]])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[3]])
        new_lane_idx = new_from_mlir_values(self._lane_idx, [values[4]])
        new_my_psum = new_from_mlir_values(self._my_psum, [values[5]])
        new_my_cluster_count_m = new_from_mlir_values(self._my_cluster_count_m, [values[6]])
        new_my_cluster_count_n = new_from_mlir_values(self._my_cluster_count_n, [values[7]])
        if cutlass.const_expr(self.use_clc):
            new_clc_response_ptr = new_from_mlir_values(self._clc_response_ptr, [values[8]])
        else:
            new_clc_response_ptr = None
        return SimpleGroupedTileScheduler(self.params, self.group_agnostic, new_linear_idx, new_cta_in_cluster_m, new_cta_in_cluster_n, new_num_tiles_executed, new_lane_idx, new_my_psum, new_my_cluster_count_m, new_my_cluster_count_n, new_clc_response_ptr)

    @staticmethod
    def compute_cluster_info(problem_sizes_m: Tuple[int, ...], problem_size_n: int, cta_tile_shape_mn: Tuple[int, int], cluster_shape_mn: Tuple[int, int], swap_mn: bool=False, cta_tile_n_per_group: Optional[Tuple[int, ...]]=None) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], int]:
        cluster_tile_shape_m = cta_tile_shape_mn[0] * cluster_shape_mn[0]
        cluster_counts_m = []
        cluster_counts_n = []
        prefix_sum = []
        running_sum = 0
        for idx, var_size in enumerate(problem_sizes_m):
            if swap_mn:
                m_size = problem_size_n
                n_size = var_size
            else:
                m_size = var_size
                n_size = problem_size_n
            if cta_tile_n_per_group is not None:
                cluster_tile_n = cta_tile_n_per_group[idx] * cluster_shape_mn[1]
            else:
                cluster_tile_n = cta_tile_shape_mn[1] * cluster_shape_mn[1]
            count_m = ceil_div(m_size, cluster_tile_shape_m)
            count_n = ceil_div(n_size, cluster_tile_n)
            cluster_counts_m.append(count_m)
            cluster_counts_n.append(count_n)
            running_sum += count_m * count_n
            prefix_sum.append(running_sum)
        return (tuple(prefix_sum), tuple(cluster_counts_m), tuple(cluster_counts_n), running_sum)

    @staticmethod
    def get_grid_shape(total_clusters: int, cluster_shape_mn: Tuple[int, int], max_active_clusters: int) -> Tuple[int, int, int]:
        num_active = min(total_clusters, max_active_clusters)
        return (cluster_shape_mn[0], cluster_shape_mn[1], num_active)

    @staticmethod
    @cute.jit
    def create(params: TileSchedulerParams, group_agnostic: cutlass.Constexpr[bool], block_idx: Tuple[Uint32, Uint32, Uint32], clc_response_ptr: Optional[cute.Pointer]=None) -> 'SimpleGroupedTileScheduler':
        cluster_prefix_sum = params.cluster_prefix_sum
        cluster_counts_m = params.cluster_counts_m
        cluster_counts_n = params.cluster_counts_n
        total_clusters = params.total_clusters
        cluster_shape_mn = params.cluster_shape_mn
        num_groups = params.num_groups
        bidx, bidy, bidz = block_idx
        linear_idx = DTYPE(bidz)
        cta_in_cluster_m = DTYPE(bidx % cluster_shape_mn[0])
        cta_in_cluster_n = DTYPE(bidy % cluster_shape_mn[1])
        num_tiles_executed = DTYPE(0)
        lookup_mode = params.lookup_mode
        lane_idx = DTYPE(0)
        my_psum = DTYPE(total_clusters)
        my_cluster_count_m = DTYPE(1)
        my_cluster_count_n = DTYPE(1)
        if cutlass.const_expr(lookup_mode == LookupMode.POPCOUNT):
            lane_idx = cute.arch.lane_idx()
            for g in cutlass.range_constexpr(num_groups):
                is_my_lane = lane_idx == g
                psum_g = DTYPE(cluster_prefix_sum[g])
                count_m_g = DTYPE(cluster_counts_m[g])
                count_n_g = DTYPE(cluster_counts_n[g])
                my_psum = DTYPE(cutlass.select_(is_my_lane, psum_g, my_psum))
                my_cluster_count_m = DTYPE(cutlass.select_(is_my_lane, count_m_g, my_cluster_count_m))
                my_cluster_count_n = DTYPE(cutlass.select_(is_my_lane, count_n_g, my_cluster_count_n))
        return SimpleGroupedTileScheduler(params, group_agnostic, linear_idx, cta_in_cluster_m, cta_in_cluster_n, num_tiles_executed, lane_idx, my_psum, my_cluster_count_m, my_cluster_count_n, clc_response_ptr)

    @cute.jit
    def _get_current_work_for_linear_idx(self, linear_idx: Uint32) -> SimpleWorkTileInfo:
        is_valid = linear_idx < self.total_clusters
        if cutlass.const_expr(self.group_agnostic):
            return SimpleWorkTileInfo(group_idx=DTYPE(0), cta_tile_m=DTYPE(0), cta_tile_n=DTYPE(0), is_valid=is_valid)
        group_idx = DTYPE(0)
        cluster_idx_in_group = DTYPE(0)
        cta_tile_m = DTYPE(0)
        cta_tile_n = DTYPE(0)
        if is_valid:
            cluster_m = DTYPE(0)
            cluster_n = DTYPE(0)
            lookup_mode = self.params.lookup_mode
            if cutlass.const_expr(lookup_mode == LookupMode.SINGLE_WITH_TRAILING):
                tiles_per_group = cutlass.const_expr(self.params.uniform_tiles_per_group)
                max_group = cutlass.const_expr(self.params.num_groups - 1)
                raw_group_idx = linear_idx // DTYPE(tiles_per_group)
                group_idx = DTYPE(min(raw_group_idx, DTYPE(max_group)))
                prev_psum = group_idx * DTYPE(tiles_per_group)
                cluster_idx_in_group = linear_idx - prev_psum
                if group_idx < DTYPE(max_group):
                    if cutlass.const_expr(self.raster_along_m):
                        cluster_m = cluster_idx_in_group % DTYPE(self.params.cluster_counts_m[0])
                        cluster_n = cluster_idx_in_group // DTYPE(self.params.cluster_counts_m[0])
                    else:
                        cluster_n = cluster_idx_in_group % DTYPE(self.params.cluster_counts_n[0])
                        cluster_m = cluster_idx_in_group // DTYPE(self.params.cluster_counts_n[0])
                elif cutlass.const_expr(self.raster_along_m):
                    cluster_m = cluster_idx_in_group % DTYPE(self.params.cluster_counts_m[max_group])
                    cluster_n = cluster_idx_in_group // DTYPE(self.params.cluster_counts_m[max_group])
                else:
                    cluster_n = cluster_idx_in_group % DTYPE(self.params.cluster_counts_n[max_group])
                    cluster_m = cluster_idx_in_group // DTYPE(self.params.cluster_counts_n[max_group])
            elif cutlass.const_expr(lookup_mode == LookupMode.TWO_MODE):
                uniform_total = cutlass.const_expr(self.params.uniform_total_tiles)
                uniform_count = cutlass.const_expr(self.params.uniform_group_count)
                tiles_per_group_1 = cutlass.const_expr(self.params.uniform_tiles_per_group)
                tiles_per_group_2 = cutlass.const_expr(self.params.mode2_tiles_per_group)
                if linear_idx < DTYPE(uniform_total):
                    group_idx = linear_idx // DTYPE(tiles_per_group_1)
                    cluster_idx_in_group = linear_idx % DTYPE(tiles_per_group_1)
                    if cutlass.const_expr(self.raster_along_m):
                        cluster_m = cluster_idx_in_group % DTYPE(self.params.cluster_counts_m[0])
                        cluster_n = cluster_idx_in_group // DTYPE(self.params.cluster_counts_m[0])
                    else:
                        cluster_n = cluster_idx_in_group % DTYPE(self.params.cluster_counts_n[0])
                        cluster_m = cluster_idx_in_group // DTYPE(self.params.cluster_counts_n[0])
                else:
                    offset_idx = linear_idx - DTYPE(uniform_total)
                    group_idx = DTYPE(uniform_count) + offset_idx // DTYPE(tiles_per_group_2)
                    cluster_idx_in_group = offset_idx % DTYPE(tiles_per_group_2)
                    if cutlass.const_expr(self.raster_along_m):
                        cluster_m = cluster_idx_in_group % DTYPE(self.params.cluster_counts_m[uniform_count])
                        cluster_n = cluster_idx_in_group // DTYPE(self.params.cluster_counts_m[uniform_count])
                    else:
                        cluster_n = cluster_idx_in_group % DTYPE(self.params.cluster_counts_n[uniform_count])
                        cluster_m = cluster_idx_in_group // DTYPE(self.params.cluster_counts_n[uniform_count])
            else:
                group_idx = popc_u32(vote_ballot_u32(linear_idx >= self._my_psum))
                prev_lane = group_idx - DTYPE(1)
                prev_psum = DTYPE(cute.arch.shuffle_sync(self._my_psum, prev_lane))
                if group_idx == 0:
                    prev_psum = DTYPE(0)
                cluster_idx_in_group = linear_idx - prev_psum
                if cutlass.const_expr(self.raster_along_m):
                    cluster_count_m = DTYPE(cute.arch.shuffle_sync(self._my_cluster_count_m, group_idx))
                    cluster_m = cluster_idx_in_group % cluster_count_m
                    cluster_n = cluster_idx_in_group // cluster_count_m
                else:
                    cluster_count_n = DTYPE(cute.arch.shuffle_sync(self._my_cluster_count_n, group_idx))
                    cluster_n = cluster_idx_in_group % cluster_count_n
                    cluster_m = cluster_idx_in_group // cluster_count_n
            cta_tile_m = cluster_m * self.cluster_shape_mn[0] + self._cta_in_cluster_m
            cta_tile_n = cluster_n * self.cluster_shape_mn[1] + self._cta_in_cluster_n
            for g in cutlass.range_constexpr(self.num_groups):
                if group_idx == g:
                    if cutlass.const_expr(len(self.params.split_residual_swap_per_group) > g and self.params.split_residual_swap_per_group[g]):
                        n_tiles_g = cutlass.const_expr(self.params.n_tiles_per_group[g])
                        cta_tile_n = n_tiles_g - DTYPE(1) - cta_tile_n
        return SimpleWorkTileInfo(group_idx=DTYPE(group_idx), cta_tile_m=cta_tile_m, cta_tile_n=cta_tile_n, is_valid=is_valid)

    @cute.jit
    def get_current_work(self) -> SimpleWorkTileInfo:
        if cutlass.const_expr(self.use_clc):
            m_idx, n_idx, linear_idx_clc, is_valid_clc = clusterlaunchcontrol_query_cancel(self._clc_response_ptr)
            linear_idx = DTYPE(cutlass.select_(is_valid_clc, linear_idx_clc, self.total_clusters))
        else:
            linear_idx = self._linear_idx
        return self._get_current_work_for_linear_idx(linear_idx)

    @cute.jit
    def advance_to_next_work(self, mbarrier_ptr: Optional[cute.Pointer]=None) -> 'SimpleGroupedTileScheduler':
        if cutlass.const_expr(self.use_clc):
            with cute.arch.elect_one():
                clusterlaunchcontrol_try_cancel(self._clc_response_ptr, mbarrier_ptr, multicast=True)
        else:
            self._linear_idx = self._linear_idx + DTYPE(self.num_active_clusters)
        self._num_tiles_executed = self._num_tiles_executed + DTYPE(1)
        return self

    @cute.jit
    def initial_work_tile_info(self) -> SimpleWorkTileInfo:
        return self._get_current_work_for_linear_idx(self._linear_idx)

    @property
    def num_tiles_executed(self) -> Uint32:
        return self._num_tiles_executed

def ceil_div(a, b):
    return (a + b - 1) // b

@dataclass(frozen=True)
class KernelKey:
    m_sizes: tuple[int, ...]
    n: int
    k: int

    @staticmethod
    def from_input(data: input_t) -> 'KernelKey':
        _, _, _, problem_sizes = data
        m_sizes = tuple(sorted((m for m, n, k, l in problem_sizes)))
        _, n, k, _ = problem_sizes[0]
        return KernelKey(m_sizes=m_sizes, n=n, k=k)

@dataclass(frozen=True)
class CachePolicyConfig:
    long_dim_EF_all_tiles: bool = False
    long_dim_EF_single_slice_only: bool = True
    short_dim_evict_last: bool = False

@dataclass(frozen=True)
class KernelConfig:
    cluster_shape_mn: Tuple[int, int] = (CLUSTER_M, CLUSTER_N)
    mma_tiler_mn: Tuple[int, int] = (M_TILER, N_TILER)
    is_2cta: bool = IS_2CTA
    swap_mn: bool = True
    mma_tiler_per_group: bool = True
    clc_sched: bool = False
    cache_policy: Optional[CachePolicyConfig] = None
    split_residual: bool = True
    sf_tma_tensor: bool = True
    group_order: Optional[Tuple[int, ...]] = None
    split_residual_swap: Optional[Tuple[bool, ...]] = None
    prefetch_dist: Optional[int] = 0
    optim_epi: bool = False
_BASE_KERNEL_CONFIGS: dict[KernelKey, KernelConfig] = {KernelKey(m_sizes=(128, 384), n=4096, k=1536): KernelConfig(cluster_shape_mn=(4, 1), mma_tiler_mn=(256, 128), is_2cta=True, prefetch_dist=None, optim_epi=True), KernelKey(m_sizes=(192, 320), n=3072, k=4096): KernelConfig(cluster_shape_mn=(1, 1), mma_tiler_mn=(128, 128), prefetch_dist=None, is_2cta=False), KernelKey(m_sizes=(64, 72, 80, 96, 128, 160, 176, 248), n=4096, k=7168): KernelConfig(cluster_shape_mn=(2, 1), mma_tiler_mn=(256, 176), is_2cta=True, optim_epi=True, cache_policy=CachePolicyConfig(long_dim_EF_single_slice_only=True)), KernelKey(m_sizes=(40, 72, 76, 148, 160, 164, 168, 196), n=7168, k=2048): KernelConfig(cluster_shape_mn=(4, 1), mma_tiler_mn=(256, 176), is_2cta=True, optim_epi=True, group_order=(4, 5, 2, 3, 0, 1, 7, 6), cache_policy=CachePolicyConfig(long_dim_EF_single_slice_only=True))}
KERNEL_CONFIGS: dict[KernelKey, KernelConfig] = {}
for key, config in _BASE_KERNEL_CONFIGS.items():
    sorted_key = KernelKey(m_sizes=tuple(sorted(key.m_sizes)), n=key.n, k=key.k)
    KERNEL_CONFIGS[sorted_key] = config
if DELEGATED_MBAR_INIT:
    pipeline.MbarrierArray.mbarrier_init = lambda self, *args, **kwargs: None

class GroupedGemmKernel:

    def __init__(self, config: KernelConfig, kernelkey: KernelKey, max_active_clusters: int):
        self.ab_dtype = ab_dtype
        self.sf_dtype = sf_dtype
        self.sf_container_dtype = sf_container_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = config.is_2cta
        if self.use_2cta_instrs and config.cluster_shape_mn[0] * config.cluster_shape_mn[1] < 2:
            self.cluster_shape_mn = (2, 1)
        else:
            self.cluster_shape_mn = config.cluster_shape_mn
        self.cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self.mma_tiler_per_group = config.mma_tiler_per_group
        self.split_residual = self.mma_tiler_per_group and config.split_residual
        self.mma_tiler_mn = config.mma_tiler_mn
        self.mma_tiler = (*config.mma_tiler_mn, 1)
        self.mma_stage_k = 256
        self.config = config
        self.swap_mn = config.swap_mn
        self.optim_epi = config.optim_epi
        self.cache_policy = config.cache_policy
        self.max_active_clusters = max_active_clusters
        self.sf_tma_tensor = config.sf_tma_tensor or self.mma_tiler[1] > 128
        if not self.sf_tma_tensor:
            assert not self.use_2cta_instrs, 'sf_tma_tensor not supported with 2-CTA instrs. requires manual arrive'
        if self.swap_mn:
            self.M = kernelkey.n
            self.Ns = kernelkey.m_sizes
            self.Ms = None
            self.N = None
            self.num_groups = len(self.Ns)
        else:
            self.Ms = kernelkey.m_sizes
            self.N = kernelkey.n
            self.M = None
            self.Ns = None
            self.num_groups = len(self.Ms)
        self.K = kernelkey.k
        if self.mma_tiler_per_group:
            group_n_sizes = self.Ns if self.swap_mn else [self.N] * self.num_groups
            self.effective_tiler_n_per_group: list[int] = [128 if gn > self.mma_tiler_mn[1] else self.mma_tiler_mn[1] for gn in group_n_sizes]
        else:
            self.effective_tiler_n_per_group: list[int] = [self.mma_tiler_mn[1]] * self.num_groups
        self.k_tile_cnt = ceil_div(self.K, self.mma_stage_k)
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.mma_instr_ctas = 2 if self.use_2cta_instrs else 1
        self.mma_instr = tcgen05_mma_2cta_fp4 if self.use_2cta_instrs else tcgen05_mma_1cta_fp4
        self.occupancy = 1
        self.smem_capacity = utils.get_smem_capacity_in_bytes('sm_100')
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.prefetch_dist_param = config.prefetch_dist

    def _setup_attributes(self, a_tensors: List[cute.Tensor], b_tensors: List[cute.Tensor], sfa_tensors: List[cute.Tensor], sfb_tensors: List[cute.Tensor], c_tensors: List[cute.Tensor], sfa_cp_bulk_tensors: List[cute.Tensor], sfb_cp_bulk_tensors: List[cute.Tensor]):
        self._setup_mma_layouts()
        self._setup_cluster_layouts()
        self._setup_epi_layouts()
        self._setup_smem_layouts()
        self._setup_tma_layouts(a_tensors, b_tensors, sfa_tensors, sfb_tensors, c_tensors, sfa_cp_bulk_tensors, sfb_cp_bulk_tensors)
        self._setup_tile_scheduler(self.max_active_clusters)
        self._setup_cta_attributes()

    def _setup_mma_layouts(self):
        self.sAB_base_smem_desc = tcgen05_encode_base_smem_desc(ld=16, stride=1024, swizzle=2)
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1), cute.round_up(self.mma_inst_shape_mn[1], 128))
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape_mn)
        self.mma_instr_desc = encode_tcgen05_mma_descriptor_mxf4(m_dim=self.mma_inst_shape_mn[0], n_dim=self.mma_inst_shape_mn[1])
        self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb)
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.mma_tile_instrs = 4
        self.mma_tiler = (self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1], mma_inst_shape_k * self.mma_tile_instrs)
        self.mma_tiler_sfb = (self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1], mma_inst_shape_k * self.mma_tile_instrs)
        self.tiled_mmas_per_group: list[tuple] = []
        self.tiled_mmas_sfb_per_group: list[tuple] = []
        self.mma_tilers_per_group: list[tuple] = []
        self.mma_tilers_sfb_per_group: list[tuple] = []
        self.mma_instr_descs: list[tuple] = []
        self.has_residual_per_group: list[bool] = []
        self.n_sizes_per_group: list[tuple] = []
        mma_n_rounding = 16 if self.use_2cta_instrs else 8
        if self.mma_tiler_per_group:
            if self.swap_mn:
                group_n_sizes = self.Ns
            else:
                group_n_sizes = [self.N] * self.num_groups
            unique_block_cache = {}

            def get_or_create_tiled_mma(n_size: int):
                rounded_n = cute.round_up(n_size, b=mma_n_rounding)
                if rounded_n not in unique_block_cache:
                    per_group_mma_inst_shape_mn = (self.mma_inst_shape_mn[0], rounded_n)
                    per_group_mma_inst_shape_mn_sfb = (per_group_mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1), cute.round_up(n_size, 128))
                    mma_instr_desc = encode_tcgen05_mma_descriptor_mxf4(m_dim=per_group_mma_inst_shape_mn[0], n_dim=per_group_mma_inst_shape_mn[1])
                    per_group_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, self.cta_group, per_group_mma_inst_shape_mn)
                    per_group_tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE, per_group_mma_inst_shape_mn_sfb)
                    per_group_mma_tiler = (per_group_mma_inst_shape_mn[0], per_group_mma_inst_shape_mn[1], mma_inst_shape_k * self.mma_tile_instrs)
                    per_group_mma_tiler_sfb = (per_group_mma_inst_shape_mn_sfb[0], per_group_mma_inst_shape_mn_sfb[1], mma_inst_shape_k * self.mma_tile_instrs)
                    unique_block_cache[rounded_n] = (per_group_tiled_mma, per_group_tiled_mma_sfb, per_group_mma_tiler, per_group_mma_tiler_sfb, mma_instr_desc)
                return unique_block_cache[rounded_n]
            for group_idx, group_n in enumerate(group_n_sizes):
                effective_tiler_n = self.effective_tiler_n_per_group[group_idx]
                full_n = min(group_n, effective_tiler_n)
                full_cached = get_or_create_tiled_mma(full_n)
                residual_n = group_n % effective_tiler_n if self.split_residual and group_n > effective_tiler_n else 0
                has_residual = residual_n > 0
                if has_residual:
                    residual_cached = get_or_create_tiled_mma(residual_n)
                    self.tiled_mmas_per_group.append((full_cached[0], residual_cached[0]))
                    self.tiled_mmas_sfb_per_group.append((full_cached[1], residual_cached[1]))
                    self.mma_tilers_per_group.append((full_cached[2], residual_cached[2]))
                    self.mma_tilers_sfb_per_group.append((full_cached[3], residual_cached[3]))
                    self.mma_instr_descs.append((full_cached[4], residual_cached[4]))
                    self.n_sizes_per_group.append((full_n, residual_n))
                else:
                    self.tiled_mmas_per_group.append((full_cached[0],))
                    self.tiled_mmas_sfb_per_group.append((full_cached[1],))
                    self.mma_tilers_per_group.append((full_cached[2],))
                    self.mma_tilers_sfb_per_group.append((full_cached[3],))
                    self.mma_instr_descs.append((full_cached[4],))
                    self.n_sizes_per_group.append((full_n,))
                self.has_residual_per_group.append(has_residual)
            self.num_unique_block_sizes = len(unique_block_cache)
            self.n_tiles_per_group: list[int] = []
            for group_idx, group_n in enumerate(group_n_sizes):
                effective_tiler_n = self.effective_tiler_n_per_group[group_idx]
                n_tiles = ceil_div(group_n, effective_tiler_n)
                self.n_tiles_per_group.append(n_tiles)
        else:
            self.tiled_mmas_per_group.append((self.tiled_mma,))
            self.tiled_mmas_sfb_per_group.append((self.tiled_mma_sfb,))
            self.mma_tilers_per_group.append((self.mma_tiler,))
            self.mma_tilers_sfb_per_group.append((self.mma_tiler_sfb,))
            self.has_residual_per_group.append(False)
            self.n_sizes_per_group.append((self.mma_tiler_mn[1],))
            self.num_unique_block_sizes = 1
            self.n_tiles_per_group = []

    def _setup_cluster_layouts(self):
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape), self.mma_tiler[1], self.mma_tiler[2])
        self.cta_tile_shape_mnk_sfb = (self.mma_tiler_sfb[0] // cute.size(self.tiled_mma.thr_id.shape), self.mma_tiler_sfb[1], self.mma_tiler_sfb[2])
        self.cluster_tile_shape_mnk = tuple((x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))))
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (self.tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (self.tiled_mma_sfb.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

    def _setup_epi_layouts(self):
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype)
        self.epi_subtiles_per_group: list[tuple] = []
        epi_tile_n = cute.size(self.epi_tile[1])
        if self.mma_tiler_per_group:
            for i in range(len(self.mma_tilers_per_group)):
                mma_tilers_tuple = self.mma_tilers_per_group[i]
                full_n = mma_tilers_tuple[0][1]
                full_subtiles = ceil_div(full_n, epi_tile_n)
                if self.has_residual_per_group[i]:
                    residual_n = mma_tilers_tuple[1][1]
                    residual_subtiles = ceil_div(residual_n, epi_tile_n)
                    self.epi_subtiles_per_group.append((full_subtiles, residual_subtiles))
                else:
                    self.epi_subtiles_per_group.append((full_subtiles,))
        else:
            default_n = self.mma_tiler[1]
            for _ in range(self.num_groups):
                self.epi_subtiles_per_group.append((ceil_div(default_n, epi_tile_n),))

    def _setup_smem_layouts(self):
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(self.tiled_mma, self.mma_tiler, self.ab_dtype, self.ab_dtype, self.epi_tile, self.c_dtype, self.c_layout, self.sf_dtype, self.sf_vec_size, self.smem_capacity, self.occupancy)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(self.tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(self.tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(self.tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(self.tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)
        self.b_smem_layouts_staged_per_group: list[tuple] = []
        self.sfb_smem_layouts_staged_per_group: list = []

        def make_b_smem_layout(tiled_mma_i, mma_tiler_i):
            layout_ = sm100_utils.make_smem_layout_b(tiled_mma_i, mma_tiler_i, self.ab_dtype, self.num_ab_stage)
            layout = cute.make_composed_layout(layout_.inner, 0, cute.make_layout(layout_.shape, stride=layout_.outer.stride[:-1] + self.b_smem_layout_staged.outer.stride[-1:]))
            return layout

        def make_sfb_smem_layout(mma_tiler_i):
            if self.mma_tiler[1] > 128 and mma_tiler_i[1] <= 128:
                return cute.make_layout(((((32, 4), 1), (16, 4)), 1, 4, 5), stride=((((16, 4), 0), (0, 1)), 0, 512, 4096))
            else:
                return self.sfb_smem_layout_staged
        for i in range(len(self.tiled_mmas_per_group)):
            tiled_mmas_tuple = self.tiled_mmas_per_group[i]
            mma_tilers_tuple = self.mma_tilers_per_group[i]
            full_b_layout = make_b_smem_layout(tiled_mmas_tuple[0], mma_tilers_tuple[0])
            full_sfb_layout = make_sfb_smem_layout(mma_tilers_tuple[0])
            self.sfb_smem_layouts_staged_per_group.append(full_sfb_layout)
            if self.has_residual_per_group[i]:
                residual_b_layout = make_b_smem_layout(tiled_mmas_tuple[1], mma_tilers_tuple[1])
                self.b_smem_layouts_staged_per_group.append((full_b_layout, residual_b_layout))
            else:
                self.b_smem_layouts_staged_per_group.append((full_b_layout,))
        self.smem_desc_k_offset = 32 // 16
        self.smem_desc_a_stage_offset = self.a_smem_layout_staged.outer.stride[-1] // 2 // 16
        self.smem_desc_b_stage_offset = self.b_smem_layout_staged.outer.stride[-1] // 2 // 16
        if self.prefetch_dist_param is None:
            self.prefetch_dist = self.num_ab_stage
        else:
            self.prefetch_dist = self.prefetch_dist_param
        self.prefetch_enabled = self.prefetch_dist > 0

    def _setup_tma_layouts(self, a_tensors: List[cute.Tensor], b_tensors: List[cute.Tensor], sfa_tensors: List[cute.Tensor], sfb_tensors: List[cute.Tensor], c_tensors: List[cute.Tensor], sfa_cp_bulk_tensors: List[cute.Tensor], sfb_cp_bulk_tensors: List[cute.Tensor]):
        initial_a = a_tensors[0]
        initial_b = b_tensors[0][0]
        initial_sfa = sfa_tensors[0]
        initial_sfb = sfb_tensors[0]
        initial_c = c_tensors[0]
        atom_thr_size = self.mma_instr_ctas
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        self.sfa_bulk_op = cluster_shape_to_bulk_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        self.sfa_bulk_atom = cute.make_copy_atom(self.sfa_bulk_op, self.sf_container_dtype)
        self.sfb_bulk_op = cluster_shape_to_bulk_atom_SFB(self.cluster_shape_mn, self.tiled_mma.thr_id)
        self.sfb_bulk_atom = cute.make_copy_atom(self.sfb_bulk_op, self.sf_container_dtype)
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        self.sf_cp_bulk_tiler = (1, self.mma_tile_instrs * sf_rows * 4 // sf_elems_per_container)
        tma_atoms_a = []
        tma_tensors_a = []
        tma_atoms_b: list[tuple] = []
        tma_tensors_b: list[tuple] = []
        tma_atoms_sfa = []
        tma_tensors_sfa = []
        tma_atoms_sfb: list = []
        tma_tensors_sfb: list = []
        tma_atoms_c = []
        tma_tensors_c = []
        for i in range(self.num_groups):
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(a_op, a_tensors[i], a_smem_layout, self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape)
            tma_atoms_a.append(tma_atom_a)
            tma_tensors_a.append(tma_tensor_a)
            if self.mma_tiler_per_group:
                b_atoms_tuple = []
                b_tensors_tuple = []
                b_smem_layouts_tuple = self.b_smem_layouts_staged_per_group[i]
                sfb_smem_layout_i = self.sfb_smem_layouts_staged_per_group[i]
                mma_tilers_tuple = self.mma_tilers_per_group[i]
                mma_tilers_sfb_tuple = self.mma_tilers_sfb_per_group[i]
                tiled_mmas_tuple = self.tiled_mmas_per_group[i]
                tiled_mmas_sfb_tuple = self.tiled_mmas_sfb_per_group[i]
                b_tensors_group_tuple = b_tensors[i]
                for j in range(len(mma_tilers_tuple)):
                    b_smem_layout_j = cute.slice_(b_smem_layouts_tuple[j], (None, None, None, 0))
                    mma_tiler_b_j = mma_tilers_tuple[j]
                    tiled_mma_b_j = tiled_mmas_tuple[j]
                    b_tensor_j = b_tensors_group_tuple[j]
                    tma_atom_b_j, tma_tensor_b_j = cute.nvgpu.make_tiled_tma_atom_B(b_op, b_tensor_j, b_smem_layout_j, mma_tiler_b_j, tiled_mma_b_j, self.cluster_layout_vmnk.shape)
                    b_atoms_tuple.append(tma_atom_b_j)
                    b_tensors_tuple.append(tma_tensor_b_j)
                tma_atoms_b.append(tuple(b_atoms_tuple))
                tma_tensors_b.append(tuple(b_tensors_tuple))
                if self.sf_tma_tensor:
                    sfb_smem_layout_sliced = cute.slice_(sfb_smem_layout_i, (None, None, None, 0))
                    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_tensors[i], sfb_smem_layout_sliced, mma_tilers_sfb_tuple[0], tiled_mmas_sfb_tuple[0], self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
                    tma_atoms_sfb.append(tma_atom_sfb)
                    tma_tensors_sfb.append(tma_tensor_sfb)
                else:
                    tma_atoms_sfb.append(self.sfb_bulk_atom)
                    tma_tensors_sfb.append(sfb_cp_bulk_tensors[i])
            else:
                b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
                mma_tiler_b = self.mma_tiler
                tiled_mma_b = self.tiled_mma
                tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(b_op, b_tensors[i][0], b_smem_layout, mma_tiler_b, tiled_mma_b, self.cluster_layout_vmnk.shape)
                tma_atoms_b.append((tma_atom_b,))
                tma_tensors_b.append((tma_tensor_b,))
                if self.sf_tma_tensor:
                    sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
                    mma_tiler_sfb = self.mma_tiler_sfb
                    tiled_mma_sfb = self.tiled_mma_sfb
                    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_tensors[i], sfb_smem_layout, mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
                    tma_atoms_sfb.append(tma_atom_sfb)
                    tma_tensors_sfb.append(tma_tensor_sfb)
                else:
                    tma_atoms_sfb.append(self.sfb_bulk_atom)
                    tma_tensors_sfb.append(sfb_cp_bulk_tensors[i])
            if self.sf_tma_tensor:
                tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(sfa_op, sfa_tensors[i], sfa_smem_layout, self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16)
                tma_atoms_sfa.append(tma_atom_sfa)
                tma_tensors_sfa.append(tma_tensor_sfa)
            else:
                tma_atoms_sfa.append(self.sfa_bulk_atom)
                tma_tensors_sfa.append(sfa_cp_bulk_tensors[i])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_tensors[i], epi_smem_layout, self.epi_tile)
            tma_atoms_c.append(tma_atom_c)
            tma_tensors_c.append(tma_tensor_c)
        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.num_tma_load_bytes_per_group: list[tuple] = []
        self.b_copy_size_per_group: list[tuple] = []
        self.sfb_copy_size_per_group: list = []
        for i in range(self.num_groups):
            if self.mma_tiler_per_group:
                b_smem_layouts_tuple = self.b_smem_layouts_staged_per_group[i]
                sfb_smem_layout_i = self.sfb_smem_layouts_staged_per_group[i]
                b_copy_sizes = []
                tma_load_bytes = []
                sfb_smem_layout_sliced = cute.slice_(sfb_smem_layout_i, (None, None, None, 0))
                sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout_sliced)
                self.sfb_copy_size_per_group.append(sfb_copy_size)
                for j in range(len(b_smem_layouts_tuple)):
                    b_smem_layout_j = cute.slice_(b_smem_layouts_tuple[j], (None, None, None, 0))
                    b_copy_size_j = cute.size_in_bytes(self.ab_dtype, b_smem_layout_j)
                    b_copy_sizes.append(b_copy_size_j)
                    tma_load_bytes.append((a_copy_size + b_copy_size_j + sfa_copy_size + sfb_copy_size) * atom_thr_size)
                self.b_copy_size_per_group.append(tuple(b_copy_sizes))
                self.num_tma_load_bytes_per_group.append(tuple(tma_load_bytes))
            else:
                b_smem_layout_i = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
                sfb_smem_layout_i = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
                b_copy_size_i = cute.size_in_bytes(self.ab_dtype, b_smem_layout_i)
                sfb_copy_size_i = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout_i)
                self.b_copy_size_per_group.append((b_copy_size_i,))
                self.sfb_copy_size_per_group.append(sfb_copy_size_i)
                self.num_tma_load_bytes_per_group.append(((a_copy_size + b_copy_size_i + sfa_copy_size + sfb_copy_size_i) * atom_thr_size,))
        self.tma_atoms_a = tma_atoms_a
        self.tma_tensors_a = tma_tensors_a
        self.tma_atoms_b = tma_atoms_b
        self.tma_tensors_b = tma_tensors_b
        self.tma_atoms_sfa = tma_atoms_sfa
        self.tma_tensors_sfa = tma_tensors_sfa
        self.tma_atoms_sfb = tma_atoms_sfb
        self.tma_tensors_sfb = tma_tensors_sfb
        self.tma_atoms_c = tma_atoms_c
        self.tma_tensors_c = tma_tensors_c

    def _setup_tile_scheduler(self, max_active_clusters: int) -> None:
        if cutlass.const_expr(self.swap_mn):
            problem_sizes_m = self.Ns
            problem_size_n = self.M
            total_m = problem_size_n
            total_n = sum(problem_sizes_m)
        else:
            problem_sizes_m = self.Ms
            problem_size_n = self.N
            total_m = sum(problem_sizes_m)
            total_n = problem_size_n
        raster_along_m = total_m < total_n
        cta_tile_shape_mn = (128, self.mma_tiler_mn[1])
        cta_tile_n_per_group = tuple(self.effective_tiler_n_per_group)
        cluster_prefix_sum, cluster_counts_m, cluster_counts_n, total_clusters, self.grid_dim, self.clc_sched = self._compute_grid(problem_sizes_m, problem_size_n, cta_tile_shape_mn, self.cluster_shape_mn, max_active_clusters, self.swap_mn, clc_sched=self.config.clc_sched, cta_tile_n_per_group=cta_tile_n_per_group)
        num_active_clusters = min(total_clusters, max_active_clusters)
        lookup_mode, uniform_tiles_per_group, uniform_group_count, uniform_total_tiles, mode2_tiles_per_group, first_iter_can_use_divide = TileSchedulerParams.compute_lookup_mode(cluster_prefix_sum, self.num_groups, num_active_clusters)
        if self.config.split_residual_swap is not None:
            split_residual_swap_per_group = self.config.split_residual_swap
        else:
            split_residual_swap_per_group = tuple((False for _ in range(self.num_groups)))
        self.tile_sched_params = TileSchedulerParams(cluster_prefix_sum=cluster_prefix_sum, cluster_counts_m=cluster_counts_m, cluster_counts_n=cluster_counts_n, total_clusters=total_clusters, num_active_clusters=num_active_clusters, cluster_shape_mn=self.cluster_shape_mn, raster_along_m=raster_along_m, num_groups=self.num_groups, lookup_mode=lookup_mode, uniform_tiles_per_group=uniform_tiles_per_group, uniform_group_count=uniform_group_count, uniform_total_tiles=uniform_total_tiles, mode2_tiles_per_group=mode2_tiles_per_group, first_iter_can_use_divide=first_iter_can_use_divide, use_clc=self.clc_sched, n_tiles_per_group=tuple(self.n_tiles_per_group), split_residual_swap_per_group=split_residual_swap_per_group)

    def _setup_cta_attributes(self):
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id) + ((self.sched_warp_id,) if self.clc_sched else ()))
        self.epi_store_warp = self.epilog_warp_id[0]
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)))
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(self, a_tensors: List[cute.Tensor], b_tensors: List[cute.Tensor], c_tensors: List[cute.Tensor], sfa_tensors: List[cute.Tensor], sfb_tensors: List[cute.Tensor], sfa_cp_bulk_tensors: List[cute.Tensor], sfb_cp_bulk_tensors: List[cute.Tensor]):
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensors[0]).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensors[0][0]).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensors[0])
        if cutlass.const_expr(self.ab_dtype != self.ab_dtype):
            raise TypeError(f'Type mismatch: {self.ab_dtype} != {self.ab_dtype}')
        self._setup_attributes(a_tensors, b_tensors, sfa_tensors, sfb_tensors, c_tensors, sfa_cp_bulk_tensors, sfb_cp_bulk_tensors)
        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = self.cta_tile_shape_mnk[0] // sf_atom_mn * self.mma_tile_instrs
        self.num_sfb_tmem_cols = self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn * self.mma_tile_instrs
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            if cutlass.const_expr(self.clc_sched):
                clc_mbar_ptr: cute.struct.MemRange[(cutlass.Int64, self.num_clc_stage * 2)]
                clc_response_ptr: cute.struct.MemRange[cutlass.Int32, 4]
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)], self.buffer_align_bytes]
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)], self.buffer_align_bytes]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], self.buffer_align_bytes]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], self.buffer_align_bytes]
        self.shared_storage = SharedStorage
        kernel = self.kernel(self.tiled_mma, self.tiled_mmas_per_group, self.tiled_mmas_sfb_per_group, self.tma_atoms_a, self.tma_tensors_a, self.tma_atoms_b, self.tma_tensors_b, self.tma_atoms_sfa, self.tma_tensors_sfa, self.tma_atoms_sfb, self.tma_tensors_sfb, self.tma_atoms_c, self.tma_tensors_c, self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk, self.a_smem_layout_staged, self.b_smem_layout_staged, self.b_smem_layouts_staged_per_group, self.sfa_smem_layout_staged, self.sfb_smem_layout_staged, self.sfb_smem_layouts_staged_per_group, self.c_smem_layout_staged, self.epi_tile)
        kernel.launch(grid=self.grid_dim, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1), smem=self.shared_storage.size_in_bytes(), min_blocks_per_mp=1)
        return

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tiled_mmas: List[cute.TiledMma], tiled_mmas_sfb: List[cute.TiledMma], tma_atoms_a: List[cute.CopyAtom], mA_mkl: List[cute.Tensor], tma_atoms_b: List[cute.CopyAtom], mB_nkl: List[cute.Tensor], tma_atoms_sfa: list[cute.CopyAtom], mSFA_mkl: list[cute.Tensor], tma_atoms_sfb: list[cute.CopyAtom], mSFB_nkl: list[cute.Tensor], tma_atoms_c: List[cute.CopyAtom], mC_mnl: List[cute.Tensor], cluster_layout_vmnk: cute.Layout, cluster_layout_sfb_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, b_smem_layouts_staged_per_group: List[cute.ComposedLayout], sfa_smem_layout_staged: cute.Layout, sfb_smem_layout_staged: cute.Layout, sfb_smem_layouts_staged_per_group: List[cute.Layout], c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout], epi_tile: cute.Tile):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        self.evict_first = cute.Int64(cute.Int64(1364590687093260288).ir_value())
        self.evict_last = cute.Int64(cute.Int64(1508705875169116160).ir_value())
        for mma_desc_tuple in self.mma_instr_descs:
            for mma_desc in mma_desc_tuple:
                assert cute.is_static(mma_desc)
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.mma_instr_ctas
        is_leader_cta = mma_tile_coord_v == 0
        if cutlass.const_expr(self.cluster_size > 1):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        else:
            cta_rank_in_cluster = 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf
        if cutlass.const_expr(self.cluster_size > 1):
            cluster_layout_vmnk_pipeline = cluster_layout_vmnk
        else:
            cluster_layout_vmnk_pipeline = None
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(barrier_storage=storage.ab_full_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage, producer_group=ab_pipeline_producer_group, consumer_group=ab_pipeline_consumer_group, tx_count=self.num_tma_load_bytes_per_group[0][0], cta_layout_vmnk=cluster_layout_vmnk_pipeline, defer_sync=True)
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if self.use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.acc_full_mbar_ptr.data_ptr(), num_stages=self.num_acc_stage, producer_group=acc_pipeline_producer_group, consumer_group=acc_pipeline_consumer_group, cta_layout_vmnk=cluster_layout_vmnk_pipeline, defer_sync=True)
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None
        if cutlass.const_expr(self.clc_sched):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(self.cluster_shape_mn)
            num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + 1 + len(self.epilog_warp_id)))
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_clc_consumer_threads)
            clc_pipeline = PipelineClcFetchAsync.create(barrier_storage=storage.clc_mbar_ptr.data_ptr(), num_stages=self.num_clc_stage, producer_group=clc_pipeline_producer_group, consumer_group=clc_pipeline_consumer_group, tx_count=self.num_clc_response_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
            clc_response_ptr = storage.clc_response_ptr.data_ptr()
            clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_clc_stage)
        if cutlass.const_expr(DELEGATED_MBAR_INIT):
            if warp_idx == self.tma_warp_id:
                ab_empty_sync: pipeline.MbarrierArray = ab_pipeline.sync_object_empty
                for i in cutlass.range_constexpr(self.num_ab_stage):
                    cute.arch.mbarrier_init(ab_empty_sync.get_barrier(i), ab_empty_sync.arrive_count)
            if warp_idx == self.mma_warp_id:
                ab_full_sync: pipeline.MbarrierArray = ab_pipeline.sync_object_full
                for i in cutlass.range_constexpr(self.num_ab_stage):
                    cute.arch.mbarrier_init(ab_full_sync.get_barrier(i), ab_full_sync.arrive_count)
            if cutlass.const_expr(self.clc_sched):
                if warp_idx == self.sched_warp_id:
                    clc_full_sync: pipeline.MbarrierArray = clc_pipeline.sync_object_full
                    clc_empty_sync: pipeline.MbarrierArray = clc_pipeline.sync_object_empty
                    for i in cutlass.range_constexpr(self.num_clc_stage):
                        cute.arch.mbarrier_init(clc_full_sync.get_barrier(i), clc_full_sync.arrive_count)
                        cute.arch.mbarrier_init(clc_empty_sync.get_barrier(i), clc_empty_sync.arrive_count)
            if warp_idx == 2:
                acc_full_sync: pipeline.MbarrierArray = acc_pipeline.sync_object_full
                for i in cutlass.range_constexpr(self.num_acc_stage):
                    cute.arch.mbarrier_init(acc_full_sync.get_barrier(i), acc_full_sync.arrive_count)
            if warp_idx == 3:
                acc_empty_sync: pipeline.MbarrierArray = acc_pipeline.sync_object_empty
                for i in cutlass.range_constexpr(self.num_acc_stage):
                    cute.arch.mbarrier_init(acc_empty_sync.get_barrier(i), acc_empty_sync.arrive_count)
        if cutlass.const_expr(self.use_2cta_instrs):
            if warp_idx == 0:
                num_tmem_dealloc_threads = 32
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfa_cta_layout = a_cta_layout
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tCgC_list = []
        tAsA_list = []
        tAgA_list = []
        tBsB_list = []
        tBgB_list = []
        tAsSFA_list = []
        tAgSFA_list = []
        tBsSFB_list = []
        tBgSFB_list = []
        sSF_cp_bulk_layout = cute.make_ordered_layout((self.sf_cp_bulk_tiler[1], self.num_ab_stage), order=(0, 1))
        for i in cutlass.range_constexpr(self.num_groups):
            if cutlass.const_expr(self.mma_tiler_per_group):
                tiled_mma_i = tiled_mmas[i][0]
                tiled_mma_sfb_i = tiled_mmas_sfb[i][0]
                mma_tiler_i = self.mma_tilers_per_group[i][0]
                mma_tiler_sfb_i = self.mma_tilers_sfb_per_group[i][0]
                if cutlass.const_expr(self.sf_tma_tensor):
                    mSFB_nkl_i = mSFB_nkl[i]
                else:
                    mSFB_nkl_i = None
            else:
                tiled_mma_i = tiled_mmas[0][0]
                tiled_mma_sfb_i = tiled_mmas_sfb[0][0]
                mma_tiler_i = self.mma_tilers_per_group[0][0]
                mma_tiler_sfb_i = self.mma_tilers_sfb_per_group[0][0]
                if cutlass.const_expr(self.sf_tma_tensor):
                    mSFB_nkl_i = mSFB_nkl[i]
                else:
                    mSFB_nkl_i = None
            thr_mma = tiled_mma_i.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb_i.get_slice(mma_tile_coord_v)
            gA_mkl = cute.local_tile(mA_mkl[i], cute.slice_(mma_tiler_i, (None, 0, None)), (None, None, None))
            if cutlass.const_expr(self.sf_tma_tensor):
                gSFA_mkl = cute.local_tile(mSFA_mkl[i], cute.slice_(mma_tiler_i, (None, 0, None)), (None, None, None))
                gSFB_nkl = cute.local_tile(mSFB_nkl_i, cute.slice_(mma_tiler_sfb_i, (0, None, None)), (None, None, None))
            else:
                gSFA_mkl = mSFA_mkl[i]
                gSFB_nkl = mSFB_nkl[i]
            gC_mnl = cute.local_tile(mC_mnl[i], cute.slice_(mma_tiler_i, (None, None, 0)), (None, None, None))
            tCgA = thr_mma.partition_A(gA_mkl)
            if cutlass.const_expr(self.sf_tma_tensor):
                tCgSFA = thr_mma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
            else:
                tCgSFA = None
                tCgSFB = None
            tCgC = thr_mma.partition_C(gC_mnl)
            tCgC_list.append(tCgC)
            tAsA, tAgA = cpasync.tma_partition(tma_atoms_a[i], block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
            tAsA_list.append(tAsA)
            tAgA_list.append(tAgA)
            if cutlass.const_expr(self.sf_tma_tensor):
                if cutlass.const_expr(self.mma_tiler_per_group):
                    sfb_smem_layout_i = sfb_smem_layouts_staged_per_group[i]
                    tma_atom_sfb_i = tma_atoms_sfb[i]
                    sSFB_i = storage.sSFB.get_tensor(sfb_smem_layout_i)
                    tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(tma_atom_sfb_i, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB_i, 0, 3), cute.group_modes(tCgSFB, 0, 3))
                else:
                    tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(tma_atoms_sfb[i], block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
                tBsSFB = cute.filter_zeros(tBsSFB)
                tBgSFB = cute.filter_zeros(tBgSFB)
                tBsSFB_list.append(tBsSFB)
                tBgSFB_list.append(tBgSFB)
            else:
                tBsSFB = cute.make_tensor(cute.recast_ptr(sSFB.iterator, dtype=self.sf_container_dtype), sSF_cp_bulk_layout)
                tBgSFB = cute.tiled_divide(gSFB_nkl, self.sf_cp_bulk_tiler)
                tBsSFB_list.append(tBsSFB)
                tBgSFB_list.append(tBgSFB)
            if cutlass.const_expr(self.sf_tma_tensor):
                tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(tma_atoms_sfa[i], block_in_cluster_coord_vmnk[2], sfa_cta_layout, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
                tAsSFA = cute.filter_zeros(tAsSFA)
                tAgSFA = cute.filter_zeros(tAgSFA)
                tAsSFA_list.append(tAsSFA)
                tAgSFA_list.append(tAgSFA)
            else:
                tAsSFA = cute.make_tensor(cute.recast_ptr(sSFA.iterator, dtype=self.sf_container_dtype), sSF_cp_bulk_layout)
                tAgSFA = cute.tiled_divide(gSFA_mkl, self.sf_cp_bulk_tiler)
                tAsSFA_list.append(tAsSFA)
                tAgSFA_list.append(tAgSFA)
        for i in cutlass.range_constexpr(len(mA_mkl)):
            mB_nkl_tuple = mB_nkl[i]
            tBsB_tuple = []
            tBgB_tuple = []
            if cutlass.const_expr(self.mma_tiler_per_group):
                mma_tilers_tuple = self.mma_tilers_per_group[i]
                tiled_mmas_tuple = tiled_mmas[i]
                b_smem_layouts_tuple = b_smem_layouts_staged_per_group[i]
                tma_atoms_b_tuple = tma_atoms_b[i]
                for j in cutlass.range_constexpr(len(mma_tilers_tuple)):
                    mma_tiler_j = mma_tilers_tuple[j]
                    tiled_mma_j = tiled_mmas_tuple[j]
                    mB_nkl_j = mB_nkl_tuple[j]
                    thr_mma_j = tiled_mma_j.get_slice(mma_tile_coord_v)
                    b_smem_layout_staged_j = b_smem_layouts_tuple[j]
                    tma_atom_b_j = tma_atoms_b_tuple[j]
                    gB_nkl_j = cute.local_tile(mB_nkl_j, cute.slice_(mma_tiler_j, (0, None, None)), (None, None, None))
                    tCgB_j = thr_mma_j.partition_B(gB_nkl_j)
                    sB_j = storage.sB.get_tensor(b_smem_layout_staged_j.outer, swizzle=b_smem_layout_staged_j.inner)
                    tBsB_j, tBgB_j = cpasync.tma_partition(tma_atom_b_j, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB_j, 0, 3), cute.group_modes(tCgB_j, 0, 3))
                    tBsB_tuple.append(tBsB_j)
                    tBgB_tuple.append(tBgB_j)
                tBsB_list.append(tuple(tBsB_tuple))
                tBgB_list.append(tuple(tBgB_tuple))
            else:
                mma_tiler_i = self.mma_tilers_per_group[0][0]
                tiled_mma_i = tiled_mmas[0][0]
                thr_mma = tiled_mma_i.get_slice(mma_tile_coord_v)
                mB_nkl_0 = mB_nkl_tuple[0]
                gB_nkl = cute.local_tile(mB_nkl_0, cute.slice_(mma_tiler_i, (0, None, None)), (None, None, None))
                tCgB = thr_mma.partition_B(gB_nkl)
                tBsB, tBgB = cpasync.tma_partition(tma_atoms_b[i][0], block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
                tBsB_list.append((tBsB,))
                tBgB_list.append((tBgB,))
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        if warp_idx == self.tma_warp_id:
            self.warp_tma(tma_atoms_a, tma_atoms_b, tma_atoms_sfa, tma_atoms_sfb, ab_pipeline, ab_full_mbar_ptr, tAgA_list, tBgB_list, tAgSFA_list, tBgSFB_list, tAsA_list, tBsB_list, tAsSFA_list, tBsSFB_list, a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, clc_pipeline, clc_response_ptr, clc_consumer_state)
        if cutlass.const_expr(self.clc_sched):
            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                self.warp_sched(clc_pipeline, clc_response_ptr)
        if warp_idx == self.mma_warp_id:
            self.warp_mma(tiled_mma, ab_pipeline, acc_pipeline, tCtAcc_fake, sfa_smem_layout_staged, sfb_smem_layout_staged, sSFA, sSFB, tCrA, tCrB, sA, sB, is_leader_cta, clc_pipeline, clc_response_ptr, clc_consumer_state)
        if warp_idx < self.mma_warp_id:
            if cutlass.const_expr(self.optim_epi):
                self.warp_epi_optim(warp_idx, tidx, tma_atoms_c, acc_pipeline, tmem_holding_buf, tmem_dealloc_mbar_ptr, tCtAcc_fake, tCgC_list, sC, epi_tile, self.use_2cta_instrs, cta_rank_in_cluster, clc_pipeline, clc_response_ptr, clc_consumer_state)
            else:
                self.warp_epi(warp_idx, tidx, tma_atoms_c, acc_pipeline, tmem_holding_buf, tmem_dealloc_mbar_ptr, tCtAcc_fake, tCgC_list, sC, epi_tile, self.use_2cta_instrs, cta_rank_in_cluster, clc_pipeline, clc_response_ptr, clc_consumer_state)

    @cute.jit
    def _dispatch_tma_group(self, g: cutlass.Constexpr[int], ab_prod, cta_tile_info, tiled_mma: cute.TiledMma, tma_atoms_a: List[cute.CopyAtom], tma_atoms_b: List[cute.CopyAtom], tma_atoms_sfa: List[cute.CopyAtom], tma_atoms_sfb: List[cute.CopyAtom], tAgA_list: List[cute.Tensor], tBgB_list: List[cute.Tensor], tAgSFA_list: List[cute.Tensor], tBgSFB_list: List[cute.Tensor], tAsA_list: List[cute.Tensor], tBsB_list: List[cute.Tensor], tAsSFA_list: List[cute.Tensor], tBsSFB_list: List[cute.Tensor], a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask):
        if cutlass.const_expr(g == self.num_groups):
            ab_prod = ab_prod
        elif g == cta_tile_info.group_idx:
            ab_prod = self.step_tma(tiled_mma, tma_atoms_a[g], tma_atoms_b[g], tma_atoms_sfa[g], tma_atoms_sfb[g], ab_prod, tAgA_list[g], tBgB_list[g], tAgSFA_list[g], tBgSFB_list[g], tAsA_list[g], tBsB_list[g], tAsSFA_list[g], tBsSFB_list[g], a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, cta_tile_info)
        else:
            ab_prod = self._dispatch_tma_group(g + 1, ab_prod, cta_tile_info, tiled_mma, tma_atoms_a, tma_atoms_b, tma_atoms_sfa, tma_atoms_sfb, tAgA_list, tBgB_list, tAgSFA_list, tBgSFB_list, tAsA_list, tBsB_list, tAsSFA_list, tBsSFB_list, a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask)
        return ab_prod

    @cute.jit
    def warp_sched(self, clc_pipeline: PipelineClcFetchAsync, clc_response_ptr: cute.Pointer):
        tile_sched = SimpleGroupedTileScheduler.create(self.tile_sched_params, True, cute.arch.block_idx(), clc_response_ptr)
        work_tile = tile_sched.initial_work_tile_info()
        clc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_clc_stage)
        clc_sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_clc_stage)
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
            tile_sched = tile_sched.advance_to_next_work(mbarrier_addr)
            clc_producer_state.advance()
            clc_pipeline.consumer_wait(clc_sched_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_sched_consumer_state)
            clc_sched_consumer_state.advance()
        clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def warp_tma(self, tma_atoms_a: List[cute.CopyAtom], tma_atoms_b: List[cute.CopyAtom], tma_atoms_sfa: List[cute.CopyAtom], tma_atoms_sfb: List[cute.CopyAtom], ab_pipeline: pipeline.PipelineTmaUmma, ab_full_mbar_ptr: cute.Pointer, tAgA_list: List[cute.Tensor], tBgB_list: List[cute.Tensor], tAgSFA_list: List[cute.Tensor], tBgSFB_list: List[cute.Tensor], tAsA_list: List[cute.Tensor], tBsB_list: List[cute.Tensor], tAsSFA_list: List[cute.Tensor], tBsSFB_list: List[cute.Tensor], a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, clc_pipeline, clc_response_ptr, clc_consumer_state):
        tile_sched = SimpleGroupedTileScheduler.create(self.tile_sched_params, False, cute.arch.block_idx(), clc_response_ptr)
        work_tile = tile_sched.initial_work_tile_info()
        ab_producer = ab_pipeline.make_producer()
        while work_tile.is_valid_tile:
            cur_group_idx = work_tile.group_idx
            for g in cutlass.range_constexpr(self.num_groups):
                if cur_group_idx == g:
                    sync_object_full: pipeline.MbarrierArray = ab_pipeline.sync_object_full
                    if cutlass.const_expr(self.cache_policy is not None):
                        if cutlass.const_expr(self.swap_mn):
                            A_cache_policy = self.evict_last if self.cache_policy.short_dim_evict_last else None
                            if cutlass.const_expr(self.cache_policy.long_dim_EF_all_tiles):
                                B_cache_policy = self.evict_first
                            elif cutlass.const_expr(self.cache_policy.long_dim_EF_single_slice_only):
                                B_cache_policy = self.evict_first if self.mma_tilers_per_group[g][0][0] >= self.M else None
                            else:
                                B_cache_policy = None
                        else:
                            if cutlass.const_expr(self.cache_policy.long_dim_EF_all_tiles):
                                A_cache_policy = self.evict_first
                            elif cutlass.const_expr(self.cache_policy.long_dim_EF_single_slice_only):
                                A_cache_policy = self.evict_first if self.mma_tilers_per_group[g][0][1] >= self.N else None
                            else:
                                A_cache_policy = None
                            B_cache_policy = self.evict_last if self.cache_policy.short_dim_evict_last else None
                    else:
                        A_cache_policy = None
                        B_cache_policy = None
                    if cutlass.const_expr(self.split_residual and self.has_residual_per_group[g]):
                        is_residual_tile = work_tile.cta_tile_n == cutlass.const_expr(self.n_tiles_per_group[g] - 1)
                        if is_residual_tile:
                            sync_object_full.tx_count = self.num_tma_load_bytes_per_group[g][1]
                            ab_producer = self.step_tma(tma_atoms_a[g], tma_atoms_b[g][1], tma_atoms_sfa[g], tma_atoms_sfb[g], ab_producer, ab_full_mbar_ptr, tAgA_list[g], tBgB_list[g][1], tAgSFA_list[g], tBgSFB_list[g], tAsA_list[g], tBsB_list[g][1], tAsSFA_list[g], tBsSFB_list[g], a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, work_tile, A_cache_policy, B_cache_policy)
                        else:
                            sync_object_full.tx_count = self.num_tma_load_bytes_per_group[g][0]
                            ab_producer = self.step_tma(tma_atoms_a[g], tma_atoms_b[g][0], tma_atoms_sfa[g], tma_atoms_sfb[g], ab_producer, ab_full_mbar_ptr, tAgA_list[g], tBgB_list[g][0], tAgSFA_list[g], tBgSFB_list[g], tAsA_list[g], tBsB_list[g][0], tAsSFA_list[g], tBsSFB_list[g], a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, work_tile, A_cache_policy, B_cache_policy)
                    else:
                        sync_object_full.tx_count = self.num_tma_load_bytes_per_group[g][0]
                        ab_producer = self.step_tma(tma_atoms_a[g], tma_atoms_b[g][0], tma_atoms_sfa[g], tma_atoms_sfb[g], ab_producer, ab_full_mbar_ptr, tAgA_list[g], tBgB_list[g][0], tAgSFA_list[g], tBgSFB_list[g], tAsA_list[g], tBsB_list[g][0], tAsSFA_list[g], tBsSFB_list[g], a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, work_tile, A_cache_policy, B_cache_policy)
            if cutlass.const_expr(self.clc_sched):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched = tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        ab_producer.tail()

    @cute.jit
    def step_tma(self, tma_atom_a: cute.CopyAtom, tma_atom_b: cute.CopyAtom, tma_atom_sfa: cute.CopyAtom, tma_atom_sfb: cute.CopyAtom, ab_producer: pipeline.PipelineProducer, ab_full_mbar_ptr: cute.Pointer, tAgA: cute.Tensor, tBgB: cute.Tensor, tAgSFA: cute.Tensor, tBgSFB: cute.Tensor, tAsA: cute.Tensor, tBsB: cute.Tensor, tAsSFA: cute.Tensor, tBsSFB: cute.Tensor, a_full_mcast_mask, b_full_mcast_mask, sfa_full_mcast_mask, sfb_full_mcast_mask, grouped_gemm_cta_tile_info, A_cache_policy: Optional[cute.Int64]=None, B_cache_policy: Optional[cute.Int64]=None):
        mma_tile_coord_mnl = (grouped_gemm_cta_tile_info.cta_tile_idx_m // self.mma_instr_ctas, grouped_gemm_cta_tile_info.cta_tile_idx_n, 0)
        tAgA_slice = tAgA[None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]]
        tBgB_slice = tBgB[None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]]
        tAgSFA_slice = tAgSFA[None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]]
        slice_n = mma_tile_coord_mnl[1]
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
            slice_n = mma_tile_coord_mnl[1] // 2
        tBgSFB_slice = tBgSFB[None, slice_n, None, mma_tile_coord_mnl[2]]
        ab_producer.reset()
        peek_ab_empty_status = ab_producer.try_acquire()
        if cutlass.const_expr(self.prefetch_enabled):
            for pf_k_tile in cutlass.range(0, min(self.prefetch_dist, self.k_tile_cnt), unroll=1):
                cute.prefetch(tma_atom_a, tAgA_slice[None, pf_k_tile])
                cute.prefetch(tma_atom_b, tBgB_slice[None, pf_k_tile])
                if cutlass.const_expr(self.sf_tma_tensor):
                    cute.prefetch(tma_atom_sfa, tAgSFA_slice[None, pf_k_tile])
                    cute.prefetch(tma_atom_sfb, tBgSFB_slice[None, pf_k_tile])
        for k_tile in cutlass.range(0, self.k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
            cute.copy(tma_atom_a, tAgA_slice[None, handle.count], tAsA[None, handle.index], tma_bar_ptr=ab_full_mbar_ptr + handle.index, mcast_mask=a_full_mcast_mask, cache_policy=A_cache_policy)
            cute.copy(tma_atom_b, tBgB_slice[None, handle.count], tBsB[None, handle.index], tma_bar_ptr=ab_full_mbar_ptr + handle.index, mcast_mask=b_full_mcast_mask, cache_policy=B_cache_policy)
            if cutlass.const_expr(self.sf_tma_tensor):
                cute.copy(tma_atom_sfa, tAgSFA_slice[None, handle.count], tAsSFA[None, handle.index], tma_bar_ptr=ab_full_mbar_ptr + handle.index, mcast_mask=sfa_full_mcast_mask, cache_policy=A_cache_policy)
                cute.copy(tma_atom_sfb, tBgSFB_slice[None, handle.count], tBsSFB[None, handle.index], tma_bar_ptr=ab_full_mbar_ptr + handle.index, mcast_mask=sfb_full_mcast_mask, cache_policy=B_cache_policy)
            else:
                with cute.arch.elect_one():
                    cp_async_bulk_g2s(tAgSFA_slice[None, handle.count].iterator, tAsSFA[None, handle.index].iterator, handle.barrier, 2048)
                    cp_async_bulk_g2s(tBgSFB_slice[None, handle.count].iterator, tBsSFB[None, handle.index].iterator, handle.barrier, 2048)
            if cutlass.const_expr(self.prefetch_enabled):
                if k_tile < self.k_tile_cnt - self.prefetch_dist:
                    future_k_tile = handle.count + self.prefetch_dist
                    cute.prefetch(tma_atom_a, tAgA_slice[None, future_k_tile])
                    cute.prefetch(tma_atom_b, tBgB_slice[None, future_k_tile])
                    if cutlass.const_expr(self.sf_tma_tensor):
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[None, future_k_tile])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[None, future_k_tile])
            peek_ab_empty_status = cutlass.Boolean(1)
            if handle.count + 1 < self.k_tile_cnt:
                peek_ab_empty_status = ab_producer.try_acquire()
        return ab_producer

    @cute.jit
    def warp_mma(self, tiled_mma: cute.TiledMma, ab_pipeline: pipeline.PipelineTmaUmma, acc_pipeline: pipeline.PipelineUmmaAsync, tCtAcc_fake: cute.Tensor, sfa_smem_layout_staged: cute.Layout, sfb_smem_layout_staged: cute.Layout, sSFA: cute.Tensor, sSFB: cute.Tensor, tCrA: cute.Tensor, tCrB: cute.Tensor, sA: cute.Tensor, sB: cute.Tensor, is_leader_cta, clc_pipeline, clc_response_ptr, clc_consumer_state):
        self.tmem_alloc_barrier.arrive_and_wait()
        acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=self.sf_dtype)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
        tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
        sA_base_desc = tcgen05_encode_smem_addr_in_desc(self.sAB_base_smem_desc, sA.iterator)
        sB_base_desc = tcgen05_encode_smem_addr_in_desc(self.sAB_base_smem_desc, sB.iterator)
        if cutlass.const_expr(self.mma_tiler_per_group):
            tile_sched = SimpleGroupedTileScheduler.create(self.tile_sched_params, False, cute.arch.block_idx(), clc_response_ptr)
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer = ab_pipeline.make_consumer()
            acc_producer = acc_pipeline.make_producer()
            while work_tile.is_valid_tile:
                cur_group_idx = work_tile.group_idx
                mma_desc = self.mma_instr_descs[0][0]
                for g in cutlass.range_constexpr(self.num_groups):
                    if cutlass.const_expr(self.has_residual_per_group[g]):
                        is_residual = work_tile.cta_tile_n == cutlass.const_expr(self.n_tiles_per_group[g] - 1)
                        group_desc = cutlass.select_(is_residual, self.mma_instr_descs[g][1], self.mma_instr_descs[g][0])
                    else:
                        group_desc = self.mma_instr_descs[g][0]
                    mma_desc = cutlass.select_(cur_group_idx == g, group_desc, mma_desc)
                if is_leader_cta:
                    ab_consumer, acc_producer = self.step_mma_asm(mma_desc, sA_base_desc, sB_base_desc, ab_consumer, acc_producer, tCtAcc_base, tCtSFA, tCtSFB, tiled_copy_s2t_sfa, tiled_copy_s2t_sfb, tCsSFA_compact_s2t, tCsSFB_compact_s2t, tCtSFA_compact_s2t, tCtSFB_compact_s2t, sB, sSFB, True, work_tile)
                else:
                    acc_producer.advance()
                if cutlass.const_expr(self.clc_sched):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched = tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
        else:
            tile_sched = SimpleGroupedTileScheduler.create(self.tile_sched_params, True, cute.arch.block_idx(), clc_response_ptr)
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer = ab_pipeline.make_consumer()
            acc_producer = acc_pipeline.make_producer()
            while work_tile.is_valid_tile:
                if is_leader_cta:
                    ab_consumer, acc_producer = self.step_mma(tiled_mma, ab_consumer, acc_producer, tCtAcc_base, tCtSFA, tCtSFB, tiled_copy_s2t_sfa, tiled_copy_s2t_sfb, tCsSFA_compact_s2t, tCsSFB_compact_s2t, tCtSFA_compact_s2t, tCtSFB_compact_s2t, tCrA, tCrB, True, work_tile)
                else:
                    acc_producer.advance()
                if cutlass.const_expr(self.clc_sched):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched = tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
        acc_producer.tail()

    @cute.jit
    def step_mma(self, tiled_mma: cute.TiledMma, ab_consumer: pipeline.PipelineConsumer, acc_producer: pipeline.PipelineProducer, tCtAcc_base: cute.Tensor, tCtSFA: cute.Tensor, tCtSFB: cute.Tensor, tiled_copy_s2t_sfa: cute.TiledCopy, tiled_copy_s2t_sfb: cute.TiledCopy, tCsSFA_compact_s2t: cute.Tensor, tCsSFB_compact_s2t: cute.Tensor, tCtSFA_compact_s2t: cute.Tensor, tCtSFB_compact_s2t: cute.Tensor, tCrA: cute.Tensor, tCrB: cute.Tensor, is_leader_cta: cute.Boolean, work_tile: SimpleWorkTileInfo):
        assert is_leader_cta, 'Only leader CTA should run MMA'
        mma_tile_coord_n = work_tile.cta_tile_idx_n
        ab_consumer.reset()
        peek_ab_full_status = ab_consumer.try_wait()
        acc_handle = acc_producer.acquire()
        tCtAcc = tCtAcc_base[None, None, None, acc_handle.index]
        tCtSFB_mma = tCtSFB
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
            offset = cutlass.Int32(mma_tile_coord_n % 2 * 2)
            shifted_ptr = cute.recast_ptr(tCtSFB.iterator + offset, dtype=self.sf_dtype)
            tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB.layout)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_tile in range(self.k_tile_cnt):
            if is_leader_cta:
                ab_handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                s2t_stage_coord = (None, None, None, None, ab_handle.index)
                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
                cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)
                for kblock_idx in cutlass.range(self.mma_tile_instrs, unroll_full=True):
                    kblock_coord = (None, None, kblock_idx, ab_handle.index)
                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_handle.release()
                peek_ab_full_status = cutlass.Boolean(1)
                if k_tile + 1 < self.k_tile_cnt:
                    peek_ab_full_status = ab_consumer.try_wait()
        acc_handle.commit()
        acc_producer.advance()
        return (ab_consumer, acc_producer)

    @cute.jit
    def step_mma_asm(self, mma_desc: cute.Int32, sA_base_desc: cute.Int64, sB_base_desc: cute.Int64, ab_consumer: pipeline.PipelineConsumer, acc_producer: pipeline.PipelineProducer, tCtAcc_base: cute.Tensor, tCtSFA: cute.Tensor, tCtSFB: cute.Tensor, tiled_copy_s2t_sfa: cute.TiledCopy, tiled_copy_s2t_sfb: cute.TiledCopy, tCsSFA_compact_s2t: cute.Tensor, tCsSFB_compact_s2t: cute.Tensor, tCtSFA_compact_s2t: cute.Tensor, tCtSFB_compact_s2t: cute.Tensor, sB: cute.Tensor, sSFB: cute.Tensor, is_leader_cta: cute.Boolean, work_tile: SimpleWorkTileInfo):
        assert is_leader_cta, 'Only leader CTA can run MMA ASM'
        mma_tile_coord_n = work_tile.cta_tile_idx_n
        ab_consumer.reset()
        peek_ab_full_status = ab_consumer.try_wait()
        acc_handle = acc_producer.acquire()
        tCtAcc = tCtAcc_base[None, None, None, acc_handle.index]
        tCtSFB_mma = tCtSFB
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            offset = cutlass.Int32(2) if mma_tile_coord_n % 2 == 1 else cutlass.Int32(0)
            shifted_ptr = cute.recast_ptr(cute.recast_ptr(tCtSFB.iterator, dtype=self.acc_dtype) + offset, dtype=self.sf_dtype)
            tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB.layout)
        elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
            offset = cutlass.Int32(mma_tile_coord_n % 2 * 2)
            shifted_ptr = cute.recast_ptr(cute.recast_ptr(tCtSFB.iterator, dtype=self.acc_dtype) + offset, dtype=self.sf_dtype)
            tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB.layout)
        pred = cute.Int32(0)
        for k_tile in range(self.k_tile_cnt):
            if is_leader_cta:
                ab_handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                stage_offset_a = cute.Int64(ab_handle.index) * self.smem_desc_a_stage_offset
                stage_offset_b = cute.Int64(ab_handle.index) * self.smem_desc_b_stage_offset
                s2t_stage_coord = (None, None, None, None, ab_handle.index)
                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
                cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)
                for kblock_idx in cutlass.range_constexpr(self.mma_tile_instrs):
                    k_offset = cute.Int64(kblock_idx) * self.smem_desc_k_offset
                    sA_desc_offset = sA_base_desc + stage_offset_a + k_offset
                    sB_desc_offset = sB_base_desc + stage_offset_b + k_offset
                    sf_kblock_coord = (None, None, kblock_idx)
                    sfa_tmem_ptr = tCtSFA[sf_kblock_coord].iterator
                    sfb_tmem_ptr = tCtSFB_mma[sf_kblock_coord].iterator
                    with cute.arch.elect_one():
                        self.mma_instr(tCtAcc.iterator, sA_desc_offset, sB_desc_offset, mma_desc, sfa_tmem_ptr, sfb_tmem_ptr, pred)
                    pred = cute.Int32(-1)
                ab_handle.release()
                peek_ab_full_status = cutlass.Boolean(1)
                if k_tile + 1 < self.k_tile_cnt:
                    peek_ab_full_status = ab_consumer.try_wait()
        acc_handle.commit()
        acc_producer.advance()
        return (ab_consumer, acc_producer)

    @cute.jit
    def warp_epi_optim(self, warp_idx, tidx, tma_atoms_c: List[cute.CopyAtom], acc_pipeline: pipeline.PipelineUmmaAsync, tmem_holding_buf, tmem_dealloc_mbar_ptr, tCtAcc_fake: cute.Tensor, tCgC_list: List[cute.Tensor], sC: cute.Tensor, epi_tile: cute.Tile, use_2cta_instrs, cta_rank_in_cluster, clc_pipeline, clc_response_ptr, clc_consumer_state):
        if warp_idx == self.epilog_warp_id[0]:
            cute.arch.alloc_tmem(self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs)
        self.tmem_alloc_barrier.arrive_and_wait()
        acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        assert tCtAcc_base.stride[-1] == self.mma_tiler[1], 'tCtAcc_base stage stride must be at == mma_tiler[1]'
        epi_tidx = tidx
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgC_list[0], epi_tile, use_2cta_instrs)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
        tma_atom_c_0, bSG_sC, bSG_gC_partitioned_0 = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atoms_c[0], tCgC_list[0], epi_tile, sC, 'group_0')
        tma_atom_c_list = [tma_atom_c_0]
        bSG_gC_partitioned_list = [bSG_gC_partitioned_0]
        for i in cutlass.range_constexpr(1, len(tma_atoms_c)):
            tma_atom_c_, _, bSG_gC_partitioned_ = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atoms_c[i], tCgC_list[i], epi_tile, sC, f'group_{i}')
            tma_atom_c_list.append(tma_atom_c_)
            bSG_gC_partitioned_list.append(bSG_gC_partitioned_)
        tile_sched = SimpleGroupedTileScheduler.create(self.tile_sched_params, False, cute.arch.block_idx(), clc_response_ptr)
        work_tile = tile_sched.initial_work_tile_info()
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
        c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
        acc_consumer = acc_pipeline.make_consumer()
        c_stage = cute.Int32(0)
        while work_tile.is_valid_tile:
            cur_group_idx = work_tile.group_idx
            for g in cutlass.range_constexpr(self.num_groups):
                if cur_group_idx == g:
                    if cutlass.const_expr(self.split_residual and self.has_residual_per_group[g]):
                        is_residual_tile = work_tile.cta_tile_n == cutlass.const_expr(self.n_tiles_per_group[g] - 1)
                        if is_residual_tile:
                            acc_consumer, c_stage = self.step_epi(warp_idx, tma_atom_c_list[g], acc_consumer, c_pipeline, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc_base, tTR_rAcc, tRS_rC, tRS_sC, bSG_sC, bSG_gC_partitioned_list[g], work_tile, self.epi_subtiles_per_group[g][1], c_stage)
                        else:
                            acc_consumer, c_stage = self.step_epi(warp_idx, tma_atom_c_list[g], acc_consumer, c_pipeline, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc_base, tTR_rAcc, tRS_rC, tRS_sC, bSG_sC, bSG_gC_partitioned_list[g], work_tile, self.epi_subtiles_per_group[g][0], c_stage)
                    else:
                        acc_consumer, c_stage = self.step_epi_optim(warp_idx, tma_atom_c_list[g], acc_consumer, c_pipeline, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc_base, tTR_rAcc, tRS_rC, tRS_sC, bSG_sC, bSG_gC_partitioned_list[g], work_tile, self.epi_subtiles_per_group[g][0], c_stage)
            if cutlass.const_expr(self.clc_sched):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched = tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        self.epilog_sync_barrier.arrive_and_wait()
        if warp_idx == self.epilog_warp_id[0]:
            if cutlass.const_expr(self.use_2cta_instrs):
                cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs)
        c_pipeline.producer_tail()

    @cute.jit
    def step_epi_optim(self, warp_idx, tma_atom_c: cute.CopyAtom, acc_consumer: pipeline.PipelineConsumer, c_pipeline: pipeline.PipelineTmaStore, tiled_copy_t2r: cute.TiledCopy, tiled_copy_r2s: cute.TiledCopy, tTR_tAcc_base: cute.Tensor, tTR_rAcc: cute.Tensor, tRS_rC: cute.Tensor, tRS_sC: cute.Tensor, bSG_sC: cute.Tensor, bSG_gC_partitioned: cute.Tensor, grouped_gemm_cta_tile_info: SimpleWorkTileInfo, epi_subtile_cnt: cutlass.Constexpr[int], c_stage: cute.Int32):
        mma_tile_coord_mnl = (grouped_gemm_cta_tile_info.cta_tile_idx_m // self.mma_instr_ctas, grouped_gemm_cta_tile_info.cta_tile_idx_n, 0)
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
        acc_handle = acc_consumer.wait()
        tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_handle.index]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        tTR_rAcc_staged = cute.make_rmem_tensor(tTR_rAcc.shape + (epi_subtile_cnt,), self.acc_dtype)
        for subtile_idx in range(epi_subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[None, None, None, subtile_idx]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc_staged[None, None, None, subtile_idx])
        for subtile_idx in range(epi_subtile_cnt):
            tTR_rAcc = tTR_rAcc_staged[None, None, None, subtile_idx]
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(acc_vec.to(self.c_dtype))
            c_buffer = c_stage % self.num_c_stage
            c_stage = c_stage + 1
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[None, None, None, c_buffer])
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epi_store_warp:
                cute.copy(tma_atom_c, bSG_sC[None, c_buffer], bSG_gC[None, subtile_idx], cache_policy=self.evict_first)
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            self.epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_handle.release()
        acc_consumer.advance()
        return (acc_consumer, c_stage)

    @cute.jit
    def warp_epi(self, warp_idx, tidx, tma_atoms_c: List[cute.CopyAtom], acc_pipeline: pipeline.PipelineUmmaAsync, tmem_holding_buf, tmem_dealloc_mbar_ptr, tCtAcc_fake: cute.Tensor, tCgC_list: List[cute.Tensor], sC: cute.Tensor, epi_tile: cute.Tile, use_2cta_instrs, cta_rank_in_cluster, clc_pipeline, clc_response_ptr, clc_consumer_state):
        if warp_idx == self.epilog_warp_id[0]:
            cute.arch.alloc_tmem(self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs)
        self.tmem_alloc_barrier.arrive_and_wait()
        acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        assert tCtAcc_base.stride[-1] == self.mma_tiler[1], 'tCtAcc_base stage stride must be at == mma_tiler[1]'
        epi_tidx = tidx
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc_base, tCgC_list[0], epi_tile, use_2cta_instrs)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
        tma_atom_c_0, bSG_sC, bSG_gC_partitioned_0 = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atoms_c[0], tCgC_list[0], epi_tile, sC, 'group_0')
        tma_atom_c_list = [tma_atom_c_0]
        bSG_gC_partitioned_list = [bSG_gC_partitioned_0]
        for i in cutlass.range_constexpr(1, len(tma_atoms_c)):
            tma_atom_c_, _, bSG_gC_partitioned_ = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atoms_c[i], tCgC_list[i], epi_tile, sC, f'group_{i}')
            tma_atom_c_list.append(tma_atom_c_)
            bSG_gC_partitioned_list.append(bSG_gC_partitioned_)
        tile_sched = SimpleGroupedTileScheduler.create(self.tile_sched_params, False, cute.arch.block_idx(), clc_response_ptr)
        work_tile = tile_sched.initial_work_tile_info()
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
        c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
        acc_consumer = acc_pipeline.make_consumer()
        c_stage = cute.Int32(0)
        while work_tile.is_valid_tile:
            cur_group_idx = work_tile.group_idx
            for g in cutlass.range_constexpr(self.num_groups):
                if cur_group_idx == g:
                    if cutlass.const_expr(self.split_residual and self.has_residual_per_group[g]):
                        is_residual_tile = work_tile.cta_tile_n == cutlass.const_expr(self.n_tiles_per_group[g] - 1)
                        epi_subtiles = cutlass.select_(is_residual_tile, self.epi_subtiles_per_group[g][1], self.epi_subtiles_per_group[g][0])
                    else:
                        epi_subtiles = self.epi_subtiles_per_group[g][0]
                    acc_consumer, c_stage = self.step_epi(warp_idx, tma_atom_c_list[g], acc_consumer, c_pipeline, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc_base, tTR_rAcc, tRS_rC, tRS_sC, bSG_sC, bSG_gC_partitioned_list[g], work_tile, epi_subtiles, c_stage)
            if cutlass.const_expr(self.clc_sched):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched = tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        self.epilog_sync_barrier.arrive_and_wait()
        if warp_idx == self.epilog_warp_id[0]:
            if cutlass.const_expr(self.use_2cta_instrs):
                cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs)
        c_pipeline.producer_tail()

    @cute.jit
    def step_epi(self, warp_idx, tma_atom_c: cute.CopyAtom, acc_consumer: pipeline.PipelineConsumer, c_pipeline: pipeline.PipelineTmaStore, tiled_copy_t2r: cute.TiledCopy, tiled_copy_r2s: cute.TiledCopy, tTR_tAcc_base: cute.Tensor, tTR_rAcc: cute.Tensor, tRS_rC: cute.Tensor, tRS_sC: cute.Tensor, bSG_sC: cute.Tensor, bSG_gC_partitioned: cute.Tensor, grouped_gemm_cta_tile_info: SimpleWorkTileInfo, epi_subtile_cnt: cutlass.Constexpr[int], c_stage: cute.Int32):
        mma_tile_coord_mnl = (grouped_gemm_cta_tile_info.cta_tile_idx_m // self.mma_instr_ctas, grouped_gemm_cta_tile_info.cta_tile_idx_n, 0)
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
        acc_handle = acc_consumer.wait()
        tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_handle.index]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        subtile_cnt = epi_subtile_cnt
        for subtile_idx in range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[None, None, None, subtile_idx]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(acc_vec.to(self.c_dtype))
            c_buffer = c_stage % self.num_c_stage
            c_stage = c_stage + 1
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[None, None, None, c_buffer])
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                cute.copy(tma_atom_c, bSG_sC[None, c_buffer], bSG_gC[None, subtile_idx], cache_policy=self.evict_first)
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            self.epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_handle.release()
        acc_consumer.advance()
        return (acc_consumer, c_stage)

    def mainloop_s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def epilog_tmem_copy_and_partition(self, tidx: cutlass.Int32, tAcc: cute.Tensor, gC_mnl: cute.Tensor, epi_tile: cute.Tile, use_2cta_instrs: Union[cutlass.Boolean, bool]) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, use_2cta_instrs)
        tAcc_epi = cute.flat_divide(tAcc[(None, None), 0, 0, None], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[None, None, 0, 0, 0])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[(None, None), 0, 0, None, None, None], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, self.acc_dtype)
        return (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r: cute.TiledCopy, tTR_rC: cute.Tensor, tidx: cutlass.Int32, sC: cute.Tensor) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return (tiled_copy_r2s, tRS_rC, tRS_sC)

    def epilog_gmem_copy_and_partition(self, tidx: cutlass.Int32, atom: Union[cute.CopyAtom, cute.TiledCopy], gC_mnl: cute.Tensor, epi_tile: cute.Tile, sC: cute.Tensor, prefix: str='') -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(gC_mnl[(None, None), 0, 0, None, None, None], epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), sC_for_tma_partition, gC_for_tma_partition)
        return (tma_atom_c, bSG_sC, bSG_gC)

    @staticmethod
    def _compute_stages(tiled_mma: cute.TiledMma, mma_tiler_mnk: Tuple[int, int, int], a_dtype: Type[cutlass.Numeric], b_dtype: Type[cutlass.Numeric], epi_tile: cute.Tile, c_dtype: Type[cutlass.Numeric], c_layout: utils.LayoutEnum, sf_dtype: Type[cutlass.Numeric], sf_vec_size: int, smem_capacity: int, occupancy: int) -> Tuple[int, int, int]:
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 2
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one) + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
        num_c_stage += (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)) // (occupancy * c_bytes_per_stage)
        return (num_acc_stage, num_ab_stage, num_c_stage)

    @staticmethod
    def _compute_grid(problem_sizes_m: tuple[int, ...], problem_size_n: int, cta_tile_shape_mn: tuple[int, int], cluster_shape_mn: tuple[int, int], max_active_clusters: int, swap_mn: bool=False, clc_sched: bool=False, cta_tile_n_per_group: Optional[tuple[int, ...]]=None) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, tuple[int, int, int], bool]:
        cluster_prefix_sum, cluster_counts_m, cluster_counts_n, total_clusters = SimpleGroupedTileScheduler.compute_cluster_info(problem_sizes_m, problem_size_n, cta_tile_shape_mn, cluster_shape_mn, swap_mn, cta_tile_n_per_group=cta_tile_n_per_group)
        static_grid_dim = SimpleGroupedTileScheduler.get_grid_shape(total_clusters, cluster_shape_mn, max_active_clusters)
        num_active_clusters = static_grid_dim[2]
        use_clc = clc_sched and total_clusters > num_active_clusters
        if use_clc:
            grid_dim = (cluster_shape_mn[0], cluster_shape_mn[1], total_clusters)
        else:
            grid_dim = static_grid_dim
        return (cluster_prefix_sum, cluster_counts_m, cluster_counts_n, total_clusters, grid_dim, use_clc)

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Tuple[int, int]) -> bool:
        is_valid = True
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [128, 256]:
            is_valid = False
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        is_power_of_2 = lambda x: x > 0 and x & x - 1 == 0
        if cluster_shape_mn[0] * cluster_shape_mn[1] > 16 or cluster_shape_mn[0] <= 0 or cluster_shape_mn[1] <= 0 or (cluster_shape_mn[0] > 4) or (cluster_shape_mn[1] > 4) or (not is_power_of_2(cluster_shape_mn[0])) or (not is_power_of_2(cluster_shape_mn[1])):
            is_valid = False
        return is_valid

    @cute.jit
    def run(self, a_ptrs: List[cute.Pointer], b_ptrs: List[cute.Pointer], sfa_ptrs: List[cute.Pointer], sfb_ptrs: List[cute.Pointer], c_ptrs: List[cute.Pointer], kernel_key: cutlass.Constexpr[KernelKey]):
        K = kernel_key.k
        if cutlass.const_expr(self.swap_mn):
            a_ptrs, b_ptrs = (b_ptrs, a_ptrs)
            sfa_ptrs, sfb_ptrs = (sfb_ptrs, sfa_ptrs)

        def make_gmem_layout(shape):
            return cute.make_ordered_layout(shape, order=(1, 0, 2))
        if cutlass.const_expr(self.swap_mn):
            make_c_gmem_layout = lambda shape: cute.make_ordered_layout(shape, order=(0, 1, 2))
        else:
            make_c_gmem_layout = lambda shape: cute.make_ordered_layout(shape, order=(1, 0, 2))
        a_tensors = []
        b_tensors = []
        c_tensors = []
        sfa_tensors = []
        sfb_tensors = []
        sfa_cp_bulk_tensors = []
        sfb_cp_bulk_tensors = []
        b_stride_n = K
        b_stride_k = 1
        for i in cutlass.range_constexpr(self.num_groups):
            if cutlass.const_expr(self.swap_mn):
                M = kernel_key.n
                N = kernel_key.m_sizes[i]
            else:
                M = kernel_key.m_sizes[i]
                N = kernel_key.n
            effective_tiler_n = cutlass.const_expr(self.effective_tiler_n_per_group[i])
            group_has_residual = cutlass.const_expr(self.split_residual and N > effective_tiler_n and (N % effective_tiler_n > 0))
            a_tensor = cute.make_tensor(a_ptrs[i], make_gmem_layout((M, K, 1)))
            a_tensors.append(a_tensor)
            if cutlass.const_expr(group_has_residual):
                b_tensor_full = cute.make_tensor(b_ptrs[i], make_gmem_layout((N, K, 1)))
                n_full_tiles = N // effective_tiler_n
                residual_n = N % effective_tiler_n
                b_offset_elements = n_full_tiles * effective_tiler_n * K
                b_ptr_residual = b_ptrs[i] + b_offset_elements
                b_layout_residual = cute.make_layout((residual_n, K, 1), stride=(b_stride_n, b_stride_k, 1))
                b_tensor_residual = cute.make_tensor(b_ptr_residual, b_layout_residual)
                b_tensors.append((b_tensor_full, b_tensor_residual))
                b_tensor_for_sfb = b_tensor_full
            else:
                b_tensor = cute.make_tensor(b_ptrs[i], make_gmem_layout((N, K, 1)))
                b_tensors.append((b_tensor,))
                b_tensor_for_sfb = b_tensor
            c_tensor = cute.make_tensor(c_ptrs[i], make_c_gmem_layout((M, N, 1)))
            c_tensors.append(c_tensor)
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
            sfa_tensor = cute.make_tensor(sfa_ptrs[i], sfa_layout)
            sfa_tensors.append(sfa_tensor)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor_for_sfb.shape, sf_vec_size)
            sfb_tensor = cute.make_tensor(sfb_ptrs[i], sfb_layout)
            sfb_tensors.append(sfb_tensor)
            sfa_cp_bulk_tensor = cute.make_tensor(cute.recast_ptr(sfa_tensor.iterator, dtype=sf_container_dtype), cute.make_ordered_layout((ceil_div(M, sf_rows), sf_rows * self.K // self.sf_vec_size // sf_elems_per_container, 1), order=(1, 0, 2)))
            sfa_cp_bulk_tensors.append(sfa_cp_bulk_tensor)
            sfb_cp_bulk_tensor = cute.make_tensor(cute.recast_ptr(sfb_tensor.iterator, dtype=sf_container_dtype), cute.make_ordered_layout((ceil_div(N, sf_rows), sf_rows * self.K // self.sf_vec_size // sf_elems_per_container, 1), order=(1, 0, 2)))
            sfb_cp_bulk_tensors.append(sfb_cp_bulk_tensor)
        self(a_tensors, b_tensors, c_tensors, sfa_tensors, sfb_tensors, sfa_cp_bulk_tensors, sfb_cp_bulk_tensors)
_MAX_ACTIVE_CLUSTERS_CACHE: dict[Tuple[int, int], int] = {}
_SM100_MAX_ACTIVE_CLUSTERS: dict[int, int] = {1: 148, 2: 74, 3: 45, 4: 33, 5: 26, 6: 22, 7: 15, 8: 15, 9: 15, 10: 11, 11: 7, 12: 7, 13: 7, 14: 7, 15: 7, 16: 7}

def _get_max_active_clusters(cluster_shape_mn: Tuple[int, int]) -> int:
    import os
    if cluster_shape_mn not in _MAX_ACTIVE_CLUSTERS_CACHE:
        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        if os.environ.get('CUTE_DSL_ARCH', '').startswith('sm_100'):
            _MAX_ACTIVE_CLUSTERS_CACHE[cluster_shape_mn] = _SM100_MAX_ACTIVE_CLUSTERS.get(cluster_size, 7)
        else:
            info = cutlass.utils.HardwareInfo()
            _MAX_ACTIVE_CLUSTERS_CACHE[cluster_shape_mn] = info.get_max_active_clusters(cluster_size)
    return _MAX_ACTIVE_CLUSTERS_CACHE[cluster_shape_mn]

def get_ptx_version_and_opt_level():
    import re
    import subprocess
    try:
        result = subprocess.run(['ptxas', '--version'], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'ptxas command failed: {e.stderr}') from e
    except FileNotFoundError as e:
        raise RuntimeError('ptxas not found. Is CUDA toolkit installed?') from e
    match = re.search('release\\s+(\\d+)\\.(\\d+)', result.stdout)
    if not match:
        raise RuntimeError(f'Failed to parse ptxas version from output: {result.stdout}')
    major, minor = (int(match.group(1)), int(match.group(2)))
    if major > 13 or (major == 13 and minor >= 1):
        return ('9.1', 'O2')
    else:
        return ('9.0', 'O3')
ptx_version, opt_level = get_ptx_version_and_opt_level()
_compiled_kernel_cache: dict[KernelKey, tuple[cutlass.cutlass_dsl.JitCompiledFunction, int]] = {}

def compile_kernel(cache_key: KernelKey):
    global _compiled_kernel_cache
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]
    sm_count = _get_max_active_clusters((1, 1))
    group_count = len(cache_key.m_sizes)
    fake_a_ptrs = [make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=32) for _ in range(group_count)]
    fake_b_ptrs = [make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=32) for _ in range(group_count)]
    fake_sfa_ptrs = [make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32) for _ in range(group_count)]
    fake_sfb_ptrs = [make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32) for _ in range(group_count)]
    fake_c_ptrs = [make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=32) for _ in range(group_count)]
    normalized_key = KernelKey(m_sizes=tuple(sorted(cache_key.m_sizes)), n=cache_key.n, k=cache_key.k)
    config = KERNEL_CONFIGS.get(normalized_key, KernelConfig())
    effective_cluster_shape_mn = config.cluster_shape_mn
    if config.is_2cta and config.cluster_shape_mn[0] * config.cluster_shape_mn[1] < 2:
        effective_cluster_shape_mn = (2, 1)
    max_active_clusters = _get_max_active_clusters(effective_cluster_shape_mn)
    kernel = GroupedGemmKernel(config, cache_key, max_active_clusters)
    rename_suffix = '_'.join((str(m) for m in cache_key.m_sizes))
    print('Compiling kernel...')
    compiled_func = cute_compile_optimized(kernel.run, fake_a_ptrs, fake_b_ptrs, fake_sfa_ptrs, fake_sfb_ptrs, fake_c_ptrs, cache_key, rename=f'group_gemm_{rename_suffix}', ptx_version=ptx_version, opt_level=opt_level, options='', apply_smem_opt=True)
    print('Kernel compilation completed.')
    _compiled_kernel_cache[cache_key] = (compiled_func, sm_count)
    return (compiled_func, sm_count)

def ref_kernel(data: input_t) -> output_t:

    def to_blocked(input_matrix):
        rows, cols = input_matrix.shape
        n_row_blocks = ceil_div(rows, 128)
        n_col_blocks = ceil_div(cols, 4)
        padded_rows = n_row_blocks * 128
        padded_cols = n_col_blocks * 4
        if padded_rows != rows or padded_cols != cols:
            padded = torch.nn.functional.pad(input_matrix, (0, padded_cols - cols, 0, padded_rows - rows), mode='constant', value=0)
        else:
            padded = input_matrix
        blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
        rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
        return rearranged.flatten()
    abc_tensors, sfasfb_tensors, _, problem_sizes = data
    result_tensors = []
    for i, ((a_ref, b_ref, c_ref), (sfa_ref, sfb_ref), (m, n, k, l)) in enumerate(zip(abc_tensors, sfasfb_tensors, problem_sizes)):
        for l_idx in range(l):
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])
            res = torch._scaled_mm(a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2), b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2), scale_a.cuda(), scale_b.cuda(), bias=None, out_dtype=torch.float16)
            c_ref[:, :, l_idx] = res
        result_tensors.append(c_ref)
    return result_tensors

def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    descending_indices = sorted(range(len(problem_sizes)), key=lambda i: -problem_sizes[i][0])
    _, n, k, _ = problem_sizes[0]
    sorted_m_sizes = tuple(sorted((problem_sizes[i][0] for i in range(len(problem_sizes)))))
    normalized_key = KernelKey(m_sizes=sorted_m_sizes, n=n, k=k)
    if normalized_key not in KERNEL_CONFIGS:
        return ref_kernel(data)
    config = KERNEL_CONFIGS.get(normalized_key, KernelConfig())
    num_groups = len(problem_sizes)
    if config.group_order is not None:
        group_order = config.group_order
    else:
        group_order = tuple(range(num_groups))
    descending_m_sizes = tuple((problem_sizes[i][0] for i in descending_indices))
    execution_m_sizes = tuple((descending_m_sizes[i] for i in group_order))
    cache_key = KernelKey(m_sizes=execution_m_sizes, n=n, k=k)
    compiled_func, sm_count = compile_kernel(cache_key)
    a_ptrs = []
    b_ptrs = []
    sfa_ptrs = []
    sfb_ptrs = []
    c_ptrs = []
    for desc_idx in group_order:
        i = descending_indices[desc_idx]
        a, b, c = abc_tensors[i]
        sfa_reordered, sfb_reordered = sfasfb_reordered_tensors[i]
        m, n, k, l = problem_sizes[i]
        a_ptrs.append(make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32))
        b_ptrs.append(make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32))
        c_ptrs.append(make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32))
        sfa_ptrs.append(make_ptr(sf_dtype, sfa_reordered.data_ptr(), cute.AddressSpace.gmem, assumed_align=32))
        sfb_ptrs.append(make_ptr(sf_dtype, sfb_reordered.data_ptr(), cute.AddressSpace.gmem, assumed_align=32))
    compiled_func(a_ptrs, b_ptrs, sfa_ptrs, sfb_ptrs, c_ptrs)
    res = [abc_tensors[i][2] for i in range(len(abc_tensors))]
    return res
