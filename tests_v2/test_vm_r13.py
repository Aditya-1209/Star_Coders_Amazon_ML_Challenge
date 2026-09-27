"""Run the launch safety contract against R13 as well as R12."""
import test_vm_r12


class R13VMTests(test_vm_r12.VMTests):
    version = 'r13'
