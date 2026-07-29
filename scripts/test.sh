#!/bin/bash
# ===================================
# A 股智能分析系统 - 测试脚本
# ===================================
#
# 使用方法：
#   ./scripts/test.sh [测试场景]
#
# 测试场景：
#   market      - 仅大盘复盘
#   a-stock     - A 股个股分析（茅台、平安银行）
#   etf         - ETF 分析
#   mixed       - A 股批量分析
#   single      - 单股推送模式
#   dry-run     - 仅获取数据不分析
#   full        - 完整流程测试
#   quick       - 快速测试（单只股票）
#   syntax      - Python 语法检查
#   all         - 运行核心离线/轻量测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

header() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}$1${NC}"
    echo "=============================================="
    echo ""
}

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_BIN=python3
    elif command -v python &> /dev/null; then
        PYTHON_BIN=python
    else
        error "Python 未安装"
        exit 1
    fi
    info "Python版本: $($PYTHON_BIN --version)"
}

check_deps() {
    info "检查 A 股依赖..."
    $PYTHON_BIN -c "import akshare" 2>/dev/null || warn "akshare 未安装，A 股在线测试可能失败"
    success "依赖检查完成"
}

test_market() {
    header "测试场景: 大盘复盘"
    $PYTHON_BIN main.py --market-review "$@"
    success "大盘复盘测试完成"
}

test_a_stock() {
    header "测试场景: A 股分析"
    $PYTHON_BIN main.py --stocks 600519,000001 --no-market-review "$@"
    success "A 股分析测试完成"
}

test_etf() {
    header "测试场景: ETF 分析"
    $PYTHON_BIN main.py --stocks 563230,512400 --no-market-review "$@"
    success "ETF 分析测试完成"
}

test_mixed() {
    header "测试场景: A 股批量分析"
    $PYTHON_BIN main.py --stocks 600519,000001,300750 --no-market-review "$@"
    success "A 股批量分析测试完成"
}

test_single() {
    header "测试场景: 单股推送模式"
    $PYTHON_BIN main.py --stocks 600519 --single-notify --no-market-review "$@"
    success "单股推送模式测试完成"
}

test_dry_run() {
    header "测试场景: Dry Run"
    $PYTHON_BIN main.py --stocks 600519 --dry-run --no-market-review --no-notify "$@"
    success "Dry Run 测试完成"
}

test_full() {
    header "测试场景: 完整流程"
    $PYTHON_BIN main.py --stocks 600519,000001 --market-review "$@"
    success "完整流程测试完成"
}

test_quick() {
    header "测试场景: 快速测试"
    $PYTHON_BIN main.py --stocks 600519 --no-market-review --no-notify "$@"
    success "快速测试完成"
}

test_syntax() {
    header "测试场景: Python 语法检查"
    $PYTHON_BIN -m compileall -q main.py src data_provider api bot
    success "语法检查完成"
}

test_flake8() {
    header "测试场景: Flake8"
    if ! command -v flake8 &> /dev/null; then
        warn "flake8 未安装，跳过"
        return 0
    fi
    flake8 src data_provider api bot main.py server.py
    success "Flake8 检查完成"
}

test_all() {
    test_syntax
    test_dry_run || warn "Dry Run 测试失败（可能是网络或配置问题）"
    success "核心测试完成"
}

main() {
    header "A 股智能分析系统 - 测试"

    check_python
    check_deps

    case "${1:-help}" in
        market) shift; test_market "$@" ;;
        a-stock|a_stock|astock) shift; test_a_stock "$@" ;;
        etf) shift; test_etf "$@" ;;
        mixed|mix) shift; test_mixed "$@" ;;
        single) shift; test_single "$@" ;;
        dry-run|dryrun|dry) shift; test_dry_run "$@" ;;
        full) shift; test_full "$@" ;;
        quick|q) shift; test_quick "$@" ;;
        syntax) shift; test_syntax "$@" ;;
        flake8|lint) shift; test_flake8 "$@" ;;
        all) shift; test_all "$@" ;;
        help|--help|-h|*)
            echo "使用方法: $0 [测试场景]"
            echo ""
            echo "测试场景:"
            echo "  market      - 仅大盘复盘"
            echo "  a-stock     - A 股个股分析"
            echo "  etf         - ETF 分析"
            echo "  mixed       - A 股批量分析"
            echo "  single      - 单股推送模式"
            echo "  dry-run     - 仅获取数据"
            echo "  full        - 完整流程"
            echo "  quick       - 快速测试"
            echo "  syntax      - 语法检查"
            echo "  flake8      - 静态检查"
            echo "  all         - 运行核心测试"
            ;;
    esac
}

main "$@"
