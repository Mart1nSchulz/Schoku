TARGET  := schoku
EXT     := cpp

XFLAGS :=
CXX := g++

# these architecture flags are required!
TARGET_ARCH := -mavx2 -mbmi

# preprocessor options
DFLAGS :=

# optimization options 
# Note: these are easily overridden on the command line, for example: make OFLAGS=-O2
OFLAGS := -O3
#OFLAGS := -O2
#OFLAGS := -g -Og
#OFLAGS := -pg -O
# asm output options
SFLAGS := -masm=intel

SRC_DIR := .
OBJ_DIR := .
DEP_DIR := .

CPPFLAGS1  = -MMD -MP -MF $(@:$(OBJ_DIR)/%.o=$(DEP_DIR)/%.d)
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
	$$(COMPILE.$(1)) $$< -S -g -o $$@ $(SFLAGS)
endef

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECT)
	$(CXX) $(LDFLAGS) $^ -o $@

$(foreach ext, $(EXT), $(eval $(call rule,$(ext))))

#$(OBJ_DIR) $(DEP_DIR):
#	mkdir -p $@

-include $(DEPEND)

clean:
	$(RM) -r $(TARGET) $(OBJECT) $(DEPEND)
