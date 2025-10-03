# This makefile will build an executable for AVX2 even on a computer that does not have AVX2.
TARGET  := schoku
EXT     := cpp

XFLAGS :=
CXX := g++

# these architecture flags are required!
TARGET_ARCH := -mavx2 -mbmi -mbmi2 -mlzcnt

# preprocessor options
DFLAGS := -DNDEBUG -DOPT_BIVALS
#DFLAGS := -DNDEBUG -DOPT_BIVALS -DOPT_SETS -DOPT_FSH -DOPT_UQR
# OPT_BIVALS
#     opportunistic bi-values processing (naked pairs and triples, Y-wings)
#     on by default, when compiled.
# OPT_SETS
#     resolves naked sets (complementary sets are labeled hidden sets
# OPT_FSH
#     resolves fishes (X-wing, sword fish and jellyfish)
# OPT_UQR
#     resolves (avoidable) unique rectangles

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

SOURCE := $(foreach ext, $(EXT), $(wildcard $(SRC_DIR)/*.$(ext)))
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
