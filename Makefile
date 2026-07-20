LATEXMK := latexmk
OPTS := -pdf -interaction=nonstopmode

TEX_DIR := docs
SRV_DIR := survey
OUT_DIR := output

# --- docs/ sources (output to docs/output/) ---
DOC_SRCS := $(filter-out $(TEX_DIR)/common.tex,$(wildcard $(TEX_DIR)/*.tex) $(wildcard $(TEX_DIR)/*/*.tex))

docname_of = $(or $(shell sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' $(2)/$(1).tex),$(1))

DOC_TGTS := $(foreach src,$(DOC_SRCS), \
  $(TEX_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(src))),$(dir $(src))).pdf)

# --- survey/ sources (output to survey/output/) ---
SRV_SRCS := $(wildcard $(SRV_DIR)/*.tex)
SRV_TGTS := $(foreach src,$(SRV_SRCS), \
  $(SRV_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(src))),$(SRV_DIR)).pdf)

ALL_TGTS := $(DOC_TGTS) $(SRV_TGTS)

.PHONY: all clean watch setup help

all: $(ALL_TGTS)

# Build rule for docs/ sources (-cd changes to source dir, outdir is absolute)
define doc_build_rule
$(TEX_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(1))),$(dir $(1))).pdf: $(1)
	@mkdir -p $(TEX_DIR)/$(OUT_DIR)
	-$(LATEXMK) $(OPTS) -outdir=$(abspath $(TEX_DIR)/$(OUT_DIR)) -cd \
	  -jobname=$(call docname_of,$(basename $(notdir $(1))),$(dir $(1))) $$<
endef

$(foreach src,$(DOC_SRCS),$(eval $(call doc_build_rule,$(src))))

# Build rule for survey/ sources (-cd changes to survey/, so outdir is relative)
define srv_build_rule
$(SRV_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(1))),$(SRV_DIR)).pdf: $(1)
	@mkdir -p $(SRV_DIR)/$(OUT_DIR)
	-$(LATEXMK) $(OPTS) -outdir=$(abspath $(SRV_DIR)/$(OUT_DIR)) -cd \
	  -jobname=$(call docname_of,$(basename $(notdir $(1))),$(SRV_DIR)) $$<
endef

$(foreach src,$(SRV_SRCS),$(eval $(call srv_build_rule,$(src))))

# Watch a file: make watch FILE=name (searches docs/, docs/*/, survey/)
watch:
	@base=$$(basename "$(FILE)" .tex); \
	dir=$$(dirname "$(FILE)" 2>/dev/null); \
	if [ "$$dir" != "." ]; then \
		src="$$dir/$$base.tex"; \
	elif [ -f "$(SRV_DIR)/$$base.tex" ]; then \
		src="$(SRV_DIR)/$$base.tex"; \
	elif [ -f "$(TEX_DIR)/$$base.tex" ]; then \
		src="$(TEX_DIR)/$$base.tex"; \
	else \
		src="$$(find $(TEX_DIR) -maxdepth 2 -name "$$base.tex" -print -quit 2>/dev/null)"; \
	fi; \
	[ -z "$$src" ] && { echo "Error: $$base.tex not found in $(TEX_DIR) or $(SRV_DIR)"; exit 1; }; \
	src_dir=$$(dirname "$$src"); \
	outdir="$$(cd "$(TEX_DIR)" && pwd)/$(OUT_DIR)"; \
	docname=$$(sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' "$$src" 2>/dev/null); \
	[ -z "$$docname" ] && docname="$$base"; \
	mkdir -p "$$outdir"; \
	$(LATEXMK) -pdf -f -pvc -outdir="$$outdir" -cd -interaction=nonstopmode \
	  -jobname="$$docname" "$$src"

# Bootstrap
setup:
	uv sync && uv run pre-commit install

# Clean
clean:
	rm -rf $(TEX_DIR)/$(OUT_DIR) $(SRV_DIR)/$(OUT_DIR)

help:
	@echo "Usage:"
	@echo "  make              Build all PDFs"
	@echo "  make watch FILE=x Watch and rebuild"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Sources in docs/   -> docs/output/"
	@echo "Sources in survey/ -> survey/output/"
