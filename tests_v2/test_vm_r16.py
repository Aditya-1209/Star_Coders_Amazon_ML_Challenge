"""Check the same detached-launch contract on the R16 VM entry point."""
import test_vm_r12


class R16VMTests(test_vm_r12.VMTests):
    version = 'r16'
