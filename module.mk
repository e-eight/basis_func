$(eval $(begin-module))

################################################################
# unit definitions
################################################################

module_units_h := basis_func
module_units_cpp-h := ho
# module_units_f :=
module_programs_cpp_test := ho_test

# module_programs_f :=
# module_generated :=

################################################################
# library creation flag
################################################################

$(eval $(library))

$(eval $(end-module))
