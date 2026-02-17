import sys
from pathlib import Path

# Add eval_suite/ to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch.cuda

torch.cuda.init()

from common.eval_base import EvalRunner, main


class DsaAttn2048EvalRunner(EvalRunner):
    use_cutlass = False
    use_batched_benchmark = False

    def get_custom_kernel(self):
        from submission import custom_kernel
        return custom_kernel

    def get_generate_input(self):
        from reference import generate_input
        return generate_input

    def get_check_implementation(self):
        from reference import check_implementation
        return check_implementation

    def get_reference_kernel(self):
        from reference import ref_kernel
        return ref_kernel

    def get_compile_kernel(self):
        try:
            from submission import compile_kernel
            return compile_kernel
        except ImportError:
            return None


if __name__ == "__main__":
    sys.exit(main(DsaAttn2048EvalRunner()))
