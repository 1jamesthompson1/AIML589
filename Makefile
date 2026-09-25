LATEXMK := latexmk
# Enable \write18 so the report can run texcount for its word count footer.
# $$$$ is needed: define-block recipes are expanded twice by make.
SHELL_ESCAPE := -e '$$$$pdflatex="pdflatex -shell-escape %O %S"'
# Plain recipes (e.g. watch) are expanded only once, so a single $$ is enough.
WATCH_SHELL_ESCAPE := -e '$$pdflatex="pdflatex -shell-escape %O %S"'
OPTS := -pdf -interaction=nonstopmode $(SHELL_ESCAPE)

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

.PHONY: all clean watch setup help wordcount artifacts-sync artifacts-pull artifacts-backup artifacts-precommit agents-docs agents-docs-stage agents-docs-check website

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
	case "$$src_dir" in \
		$(SRV_DIR)*) outdir="$$(cd "$(SRV_DIR)" && pwd)/$(OUT_DIR)" ;; \
		*) outdir="$$(cd "$(TEX_DIR)" && pwd)/$(OUT_DIR)" ;; \
	esac; \
	docname=$$(sed -n 's/^[[:space:]]*\\docname{\(.*\)}/\1/p' "$$src" 2>/dev/null); \
	[ -z "$$docname" ] && docname="$$base"; \
	mkdir -p "$$outdir"; \
	$(LATEXMK) -pdf -f -pvc -pv- $(WATCH_SHELL_ESCAPE) -outdir="$$outdir" -cd -interaction=nonstopmode \
	  -jobname="$$docname" "$$src"

# Print the report word count (same computation as the working draft notice)
wordcount:
	texcount -dir=$(TEX_DIR)/report/ -inc -sum -nobib $(TEX_DIR)/report/report.tex

# --- HF bucket artifacts (heavy outputs, kept out of git) ---
# Sync/backup logic lives here rather than in extra scripts so it stays
# colocated with the build tooling. All knobs come from .env (see
# .env.example): HF_TOKEN, HF_BUCKET, RUN_BORG_BACKUP, BORG_REPO.
# `make artifacts-precommit` is also invoked by the pre-commit hook; it is
# best-effort and never fails the commit.

# Upload artifacts/ (ft + bs mirrors) to the HF storage bucket.
artifacts-sync:
	@bash -c 'set -euo pipefail; \
	  set -a; [ -f .env ] && . ./.env; set +a; \
	  HF=$${HF_BIN:-.venv/bin/hf}; \
	  BUCKET=hf://buckets/$${HF_BUCKET:-1jamesthompson1/wvs-nz-value-alignment-evals}; \
	  exec "$$HF" buckets sync ./artifacts "$$BUCKET"'

# Download the bucket back into artifacts/ (fresh machines/checkouts).
artifacts-pull:
	@bash -c 'set -euo pipefail; \
	  set -a; [ -f .env ] && . ./.env; set +a; \
	  HF=$${HF_BIN:-.venv/bin/hf}; \
	  BUCKET=hf://buckets/$${HF_BUCKET:-1jamesthompson1/wvs-nz-value-alignment-evals}; \
	  mkdir -p artifacts; \
	  exec "$$HF" buckets sync "$$BUCKET" ./artifacts'

# Regenerate the agent reference docs (website /agents pages) from each
# profile's __init__.py docstring and situations.json. Run whenever the
# profiles change; the generated markdown is committed with the website.
agents-docs:
	uv run website/scripts/generate_agent_docs.py

# Fail (exit 1) if the committed agent docs have drifted from the profile
# source. The synced run data is ignored, so this works anywhere.
agents-docs-check:
	uv run website/scripts/generate_agent_docs.py --check

# Regenerate + re-stage the docs (pre-commit hook entrypoint): keeps a
# commit from carrying stale /agents pages when the profiles changed. The
# generated pages can drift when runs data was re-synced, so only files the
# generator actually touched get re-added.
agents-docs-stage: agents-docs
	git add -A website/src/content/agents 2>/dev/null || true

# One entry point: regenerate agent docs and serve the development site.
# Artifact synchronization is deliberately separate (artifacts-sync).
website: agents-docs
	cd website && npm run dev


# Borg backup of artifacts/ (unencrypted, auto-initialised).
artifacts-backup:
	@bash -c 'set -euo pipefail; \
	  set -a; [ -f .env ] && . ./.env; set +a; \
	  [ -d artifacts/ft ] && [ -d artifacts/bs ] || { echo "artifacts/ mirror missing"; exit 1; }; \
	  BR=$${BORG_REPO:-grid-directory/AIML589-evals-backup}; \
	  case "$$BR" in /*) ;; ~/*) BR=$$HOME/$${BR#\~/} ;; *) BR=$$HOME/$$BR ;; esac; \
	  BORG=$${BORG_BIN:-$$HOME/.local/bin/borg}; \
	  "$$BORG" info "$$BR" >/dev/null 2>&1 || "$$BORG" init --encryption=none "$$BR"; \
	  # Skip archiving when nothing in artifacts/ changed since the last one. \
	  CHANGED=$$("$$BORG" create --dry-run --list --filter=AM "$$BR::probe-$$$$-" ./artifacts 2>/dev/null | wc -l); \
	  if [ "$$CHANGED" -eq 0 ]; then \
	    echo "artifacts/ unchanged since last archive; skipping borg backup"; \
	    exit 0; \
	  fi; \
	  "$$BORG" create --stats "$$BR::$$(hostname)-$$(date +%Y%m%d-%H%M%S)" ./artifacts; \
	  "$$BORG" prune --keep-daily 7 --keep-weekly 4 --keep-monthly 6 "$$BR"'

# pre-commit hook entry: sync via artifacts-sync, then borg via
# artifacts-backup when RUN_BORG_BACKUP=true. Never fails so commits are not
# blocked by network/remote hiccups.
artifacts-precommit:
	@bash -c 'set -u; \
	  set -a; [ -f .env ] && . ./.env; set +a; \
	  echo "--- syncing artifacts/ to HF bucket ---"; \
	  $(MAKE) --no-print-directory artifacts-sync || echo "!! artifact sync failed (commit not blocked)"; \
	  echo "--- optional borg backup ---"; \
	  if [ "$${RUN_BORG_BACKUP:-false}" = "true" ]; then \
	    $(MAKE) --no-print-directory artifacts-backup || echo "!! artifact backup failed (commit not blocked)"; \
	  else \
	    echo "borg backup skipped (RUN_BORG_BACKUP != true)"; \
	  fi; \
	  exit 0'

# Bootstrap
setup:
	@[ -f .env ] || cp .env.example .env
	uv sync && uv run pre-commit install

# Clean
clean:
	rm -rf $(TEX_DIR)/$(OUT_DIR) $(SRV_DIR)/$(OUT_DIR)

help:
	@echo "Usage:"
	@echo "  make              Build all PDFs"
	@echo "  make watch FILE=x Watch and rebuild a single document"
	@echo "  make wordcount    Print the report word count (excl. references/appendices)"
	@echo "  make artifacts-sync    Upload artifacts/ to the HF bucket (does not delete)"
	@echo "  make artifacts-pull    Download the HF bucket into artifacts/"
	@echo "  make artifacts-backup  Borg backup of artifacts/ (also in pre-commit)"
	@echo "  make website          Regenerate agent docs and run the dev server"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Sources in docs/   -> docs/output/"
	@echo "Sources in survey/ -> survey/output/"
