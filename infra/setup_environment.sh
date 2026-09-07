#!/usr/bin/env bash
# Internal installer helpers shared by setup-curate and setup-train.
# Every installer uses the repo-local .venv so pipeline commands keep one path.

check_installer() {
    case "$INSTALLER" in
        pip) ;;
        uv|conda) command -v "$INSTALLER" >/dev/null 2>&1 || {
            echo "Install $INSTALLER first, or use INSTALLER=pip." >&2; return 1;
        } ;;
        *) echo "INSTALLER must be pip, uv, or conda" >&2; return 1 ;;
    esac
}

run_environment() {
    if [[ "$INSTALLER" == "conda" ]]; then
        conda run --no-capture-output --prefix "$VENV_DIR" "$@"
    else
        "$@"
    fi
}

install_packages() {
    if [[ "$INSTALLER" == "uv" ]]; then
        # The training requirements include PyTorch's CUDA index. Match pip's
        # multi-index resolution; all CUDA/framework versions remain pinned.
        uv pip install --python "$VENV_DIR/bin/python" --index-strategy unsafe-best-match "$@"
    else
        run_environment "$VENV_DIR/bin/python" -m pip install "$@"
    fi
}

install_environment() {
    local requirements="$1"
    check_installer
    if [[ "$INSTALLER" == "conda" ]]; then
        if [[ -e "$VENV_DIR" && ! -d "$VENV_DIR/conda-meta" ]]; then
            echo "$VENV_DIR is not a conda environment. Move it aside before switching installers." >&2
            return 1
        fi
        if [[ ! -d "$VENV_DIR/conda-meta" ]]; then
            conda create --prefix "$VENV_DIR" python=3.12 pip -y
        fi
    else
        if [[ -d "$VENV_DIR/conda-meta" ]]; then
            echo "$VENV_DIR is a conda environment. Use INSTALLER=conda or move it aside." >&2
            return 1
        fi
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            if [[ "$INSTALLER" == "uv" ]]; then
                uv venv --python 3.12 --seed "$VENV_DIR"
            else
                python3.12 -m venv "$VENV_DIR"
            fi
        fi
    fi
    run_environment "$VENV_DIR/bin/python" -c \
        'import sys; assert sys.version_info[:2] == (3, 12), "Setup requires Python 3.12; move the old .venv aside."'
    install_packages --upgrade pip
    install_packages -r "$requirements"
    run_environment "$VENV_DIR/bin/python" -m pip check
}

activation_command() {
    if [[ "$INSTALLER" == "conda" ]]; then
        printf 'conda activate %q' "$VENV_DIR"
    else
        printf 'source %q/bin/activate' "$VENV_DIR"
    fi
}
