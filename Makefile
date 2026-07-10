LATEXMK := latexmk
OPTS := -pdf -f -interaction=nonstopmode

TEX_DIR := docs
SRV_DIR := survey
OUT_DIR := output

# --- docs/ sources (output to docs/output/) ---
DOC_SRCS := $(filter-out $(TEX_DIR)/common.tex,$(wildcard $(TEX_DIR)/*.tex))

docname_of = $(or $(shell sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' $(2)/$(1).tex),$(1))

DOC_TGTS := $(foreach src,$(DOC_SRCS), \
  $(TEX_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(src))),$(TEX_DIR)).pdf)

# --- survey/ sources (output to survey/output/) ---
SRV_SRCS := $(wildcard $(SRV_DIR)/*.tex)
SRV_TGTS := $(foreach src,$(SRV_SRCS), \
  $(SRV_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(src))),$(SRV_DIR)).pdf)

ALL_TGTS := $(DOC_TGTS) $(SRV_TGTS)

.PHONY: all clean watch setup help

all: $(ALL_TGTS)

# Build rule for docs/ sources (-cd changes to docs/, so outdir is relative)
define doc_build_rule
$(TEX_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(1))),$(TEX_DIR)).pdf: $(1)
	@mkdir -p $(TEX_DIR)/$(OUT_DIR)
	-$(LATEXMK) $(OPTS) -outdir=$(OUT_DIR) -cd \
	  -jobname=$(call docname_of,$(basename $(notdir $(1))),$(TEX_DIR)) $$<
endef

$(foreach src,$(DOC_SRCS),$(eval $(call doc_build_rule,$(src))))

# Build rule for survey/ sources (-cd changes to survey/, so outdir is relative)
define srv_build_rule
$(SRV_DIR)/$(OUT_DIR)/$(call docname_of,$(basename $(notdir $(1))),$(SRV_DIR)).pdf: $(1)
	@mkdir -p $(SRV_DIR)/$(OUT_DIR)
	-$(LATEXMK) $(OPTS) -outdir=$(OUT_DIR) -cd \
	  -jobname=$(call docname_of,$(basename $(notdir $(1))),$(SRV_DIR)) $$<
endef

$(foreach src,$(SRV_SRCS),$(eval $(call srv_build_rule,$(src))))

# Watch a file: make watch FILE=docs/somefile (or survey/somefile)
watch:
	@dir=$$(dirname "$(FILE)" 2>/dev/null); \
	base=$$(basename "$(FILE)" .tex); \
	[ "$$dir" = "." ] && dir="$(TEX_DIR)"; \
	outdir="$$dir/$(OUT_DIR)"; \
	docname=$$(sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' "$$dir/$$base.tex" 2>/dev/null); \
	[ -z "$$docname" ] && docname="$$base"; \
	mkdir -p "$$outdir"; \
	$(LATEXMK) -pdf -f -pvc -outdir="$$outdir" -cd -interaction=nonstopmode \
	  -jobname="$$docname" "$$dir/$$base.tex"

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
