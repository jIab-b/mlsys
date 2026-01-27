import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.cuda

torch.cuda.init()

from common.eval_base import EvalRunner, main


class SparseAttentionEvalRunner(EvalRunner):
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

    def get_compile_kernel(self):
        try:
            from submission import compile_kernel
            return compile_kernel
        except ImportError:
            return None


if __name__ == "__main__":
    sys.exit(main(SparseAttentionEvalRunner()))
