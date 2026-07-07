TEX_DIR := docs
OUT_DIR := output

TEX_SRCS := $(filter-out $(TEX_DIR)/common.tex,$(wildcard $(TEX_DIR)/*.tex))

# Extract \docname{...} from a tex file, fall back to filename stem
docname = $(or $(shell sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' $(TEX_DIR)/$(1).tex),$(1))

# Build list of all PDF targets using their \docname (or filename stem)
PDF_TGTS := $(foreach src,$(basename $(notdir $(TEX_SRCS))),$(TEX_DIR)/$(OUT_DIR)/$(call docname,$(src)).pdf)

LATEXMK := latexmk
LATEXMK_OPTS := -pdf -f -outdir=$(OUT_DIR) -cd -interaction=nonstopmode

.PHONY: all clean watch setup help

all: $(PDF_TGTS)

# Generate a build rule for each tex file
# $(1) = basename of tex file (e.g., "two-pager-funding")
define tex_build_rule
$(TEX_DIR)/$(OUT_DIR)/$(call docname,$(1)).pdf: $(TEX_DIR)/$(1).tex
	-$(LATEXMK) $(LATEXMK_OPTS) -jobname=$(call docname,$(1)) $$<
endef

$(foreach src,$(basename $(notdir $(TEX_SRCS))),$(eval $(call tex_build_rule,$(src))))

# Watch a file: make watch FILE=two-pager-funding
watch:
	@docname=$$(sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' $(TEX_DIR)/$(FILE).tex 2>/dev/null); \
	[ -z "$$docname" ] && docname="$(FILE)"; \
	$(LATEXMK) -pdf -f -pvc -outdir=$(OUT_DIR) -cd -interaction=nonstopmode -jobname=$$docname $(TEX_DIR)/$(FILE).tex

# Bootstrap project: sync uv deps and install pre-commit hooks
setup:
	uv sync
	uv run pre-commit install

# Remove all build artifacts
clean:
	rm -rf $(TEX_DIR)/$(OUT_DIR)

help:
	@echo "Usage:"
	@echo "  make              Build all PDFs (output name from \\docname in each .tex)"
	@echo "  make watch FILE=x Watch and rebuild on changes"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Each .tex file can set \\docname{name} to control the output PDF name."
	@echo "Without it, the PDF is named after the .tex file."
