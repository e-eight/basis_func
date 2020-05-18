################################################################
# project name
################################################################

# This name is used for the distribution tar file.

project_name := basis_func

################################################################
# modules -- list of directories in which to search
# for module.mk include files
################################################################

# Caution: The order in which modules are declares is important, since
# it is also used in linking.  The object/library file for the
# *caller* must precede the object/library file for the *callee* for
# successful linking.

################
# programs
################

modules += .

MAKEFILE_STANDALONE = true

# legacy programs -- DEPRECATED

################
# libraries
################

# modules +=

################################################################
# extras -- list of extra files to be included
# in distribution tar file
################################################################

extras :=

################################################################
# additional project-specific make settings and rules
################################################################

# gsl
LDLIBS += -lgsl
LDLIBS += -lgslcblas
CPPFLAGS += -DHAVE_INLINE

# compile git information into executables
CPPFLAGS += -D'VCS_REVISION="$(vcs-git)"'

# basis submodule
#   map vs. hash for space lookup in basis library
CPPFLAGS += -DBASIS_HASH
