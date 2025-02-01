# This makefile will build an executable for AVX2 even on a computer that does not have AVX2.
#
# run 'make' for the default stand-alone solver
# run 'make schoku_lib.o' for a library module (without multi-threading)
# run 'make tdoku_wrapper_test' to create and test schoku_lib.o

TARGET  := schoku
EXT     := cpp

XFLAGS :=
CXX := g++

# these architecture flags are required!
TARGET_ARCH := -mavx2 -mbmi -mbmi2 -mlzcnt

# preprocessor options (OPT_TRIAD_RES is slower)
DFLAGS := -DNDEBUG
#DFLAGS := -DNDEBUG -DOPT_TRIAD_RES
# OPT_TRIAD_RES
#     resolves triads that have exactly 3 candidates

# optimization options 
# Note: these are easily overridden on the command line, for example: make OFLAGS=-O2
OFLAGS := -O3
#OFLAGS := -O2
#OFLAGS := -g -Og
# asm output options
SFLAGS := -masm=intel -fverbose-asm

# if not using build sub-dirs, leave them as '.'
SRC_DIR := .
OBJ_DIR := .
DEP_DIR := .

ifeq ($(OBJ_DIR),.)
        CPPFLAGS1  = -MMD -MP -MF $(@:%.o=$(DEP_DIR)/%.d)
else
        CPPFLAGS1  = -MMD -MP -MF $(@:$(OBJ_DIR)/%.o=$(DEP_DIR)/%.d)
endif

# -fopenmp is required to use OMP (multi-threading) - it implies -pthread
CFLAGS   := $(OFLAGS) -fopenmp -pthread -Wall -Wextra $(DFLAGS) $(XFLAGS)
# The default is -std=gnu++17 in gcc 11.4
# The gnu* flavour is required to find the ftruncate Posix function.
CXXFLAGS := -std=gnu++17 $(CFLAGS)
LDFLAGS  := -fopenmp -pthread

SOURCE := $(SRC_DIR)/schoku.cpp
OBJECT := $(SOURCE:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPEND := $(OBJECT:$(OBJ_DIR)/%.o=$(DEP_DIR)/%.d)

define rule =
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.$(1) | $(OBJ_DIR) $(DEP_DIR)
	$$(COMPILE.$(1)) $$(CPPFLAGS1) $$< -o $$@
$(OBJ_DIR)/%.s: $(SRC_DIR)/%.$(1) | $(OBJ_DIR) $(DEP_DIR)
	$$(COMPILE.$(1)) $$< -S -g -o $$@_ $(SFLAGS)
	c++filt < $$@_ > $$@
endef

.PHONY: all clean

ifeq ($(OBJ_DIR),$(DEP_DIR))
        DIRECTORIES := $(OBJ_DIR)
else
        DIRECTORIES := $(OBJ_DIR) $(DEP_DIR)
endif
all: $(TARGET)

$(TARGET):: $(OBJECT)
	$(CXX) $(LDFLAGS) $^ -o $@

$(TARGET):: $(DIRECTORIES)
	@true

$(foreach ext, $(EXT), $(eval $(call rule,$(ext))))

$(DIRECTORIES):
	mkdir -p $@

-include $(DEPEND)

clean:
	$(RM) -r $(TARGET) $(OBJECT) $(DEPEND)

schoku_lib.o: $(SOURCE)
	$(COMPILE.cpp) -DLIB_ONLY $(CPPFLAGS1) $< -o $@

tdoku_wrapper_test: schoku_lib.o tdoku_wrapper_test.o
	$(CXX) $(LDFLAGS) $^ -o $@
